"""Acceptance 3, 5, 6, 9: end-to-end runs with the fake adapter, failure modes, cancellation,
resume, determinism and cross-run isolation."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import engineering_llm_bench.catalog.registry as reg
from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import HarnessConfig, ModelConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import Profile
from engineering_llm_bench.orchestration.runner import RunRequest
from tests.conftest import FAST_TASKS, REPO, BenchEnv

pytestmark = pytest.mark.integration


def add_model(env: BenchEnv, **cfg: Any) -> str:
    model = ModelConfig.model_validate({"provider": "fake", "model_id": "fake", **cfg})
    env.models = env.models.model_copy(update={"models": [*env.models.models, model]})
    return model.alias


def run(
    env: BenchEnv, alias: str, harness: str = "agent", tasks: list[str] | None = None, **kw: Any
) -> str:
    req = RunRequest(
        model_alias=alias,
        harness=harness,
        profile=Profile.SMOKE,
        task_ids=FAST_TASKS if tasks is None else tasks,
        **kw,
    )
    return env.runner().run(req)


def attempts(env: BenchEnv, run_id: str) -> dict[str, dict[str, Any]]:
    return {a["task_id"]: a for a in env.store.attempts_for_run(run_id)}


def test_smoke_profile_completes_with_fake_reference(bench_env: BenchEnv) -> None:
    run_id = run(bench_env, "fake-reference", tasks=[])
    rows = bench_env.store.attempts_for_run(run_id)
    assert len(rows) == len(bench_env.registry.select(Profile.SMOKE))
    assert bench_env.store.get_run(run_id)["status"] == "completed"  # type: ignore[index]
    for a in rows:
        if a["status"] == "skipped":
            assert a["skip_reason"]
            continue
        assert a["status"] == "completed" and a["passed"] == 1, a
        assert a["failure_class"] == "none"
        assert a["score"] >= 90
        assert {d["dimension"] for d in bench_env.store.dimension_scores(a["id"])} >= {"functional"}
        names = {r["name"] for r in bench_env.store.artifacts(a["id"])}
        assert {"trace.jsonl", "diff.patch", "summary.md"} <= names


def test_direct_mode_reference_passes(bench_env: BenchEnv) -> None:
    rows = attempts(bench_env, run(bench_env, "fake-reference", harness="direct"))
    assert all(a["passed"] == 1 for a in rows.values())
    assert bench_env.store.get_run(rows[FAST_TASKS[0]]["run_id"])["mode"] == "direct"  # type: ignore[index]


@pytest.mark.parametrize("alias", ["fake-noop", "fake-broken"])
def test_wrong_or_missing_patch_fails_evaluator(bench_env: BenchEnv, alias: str) -> None:
    rows = attempts(bench_env, run(bench_env, alias))
    for a in rows.values():
        assert a["status"] == "completed" and a["passed"] == 0
        assert a["failure_class"] == "tests_failed"


def test_provider_error_is_classified(bench_env: BenchEnv) -> None:
    alias = add_model(bench_env, alias="fake-error", fake={"behavior": "error"})
    a = attempts(bench_env, run(bench_env, alias, tasks=FAST_TASKS[:1]))[FAST_TASKS[0]]
    assert a["failure_class"] == "provider_error" and a["passed"] == 0
    assert "fake provider configured to fail" in a["error"]


def test_provider_timeout_is_classified(bench_env: BenchEnv) -> None:
    alias = add_model(bench_env, alias="fake-slow", fake={"behavior": "timeout"}, timeout_s=1)
    a = attempts(bench_env, run(bench_env, alias, tasks=FAST_TASKS[:1]))[FAST_TASKS[0]]
    assert a["failure_class"] == "timeout"
    assert a["agent_wall_s"] < 15


def test_malformed_output_is_classified(bench_env: BenchEnv) -> None:
    alias = add_model(bench_env, alias="fake-garbage", fake={"behavior": "malformed"})
    a = attempts(bench_env, run(bench_env, alias, tasks=FAST_TASKS[:1]))[FAST_TASKS[0]]
    assert a["failure_class"] == "malformed_output" and a["malformed_outputs"] == 3
    d = attempts(bench_env, run(bench_env, alias, harness="direct", tasks=FAST_TASKS[:1]))
    assert d[FAST_TASKS[0]]["failure_class"] == "malformed_output"


def test_turn_budget_exceeded(bench_env: BenchEnv) -> None:
    profiles = dict(bench_env.bench.profiles)
    profiles[Profile.SMOKE] = profiles[Profile.SMOKE].model_copy(update={"max_turns": 2})
    bench_env.bench = bench_env.bench.model_copy(update={"profiles": profiles})
    a = attempts(bench_env, run(bench_env, "fake-reference", tasks=FAST_TASKS[:1]))[FAST_TASKS[0]]
    assert a["failure_class"] == "budget_exceeded" and a["turns"] == 2


def test_cancellation_marks_interrupted_and_resume_completes(bench_env: BenchEnv) -> None:
    alias = add_model(
        bench_env, alias="fake-latent", fake={"behavior": "reference", "latency_s": 0.4}
    )
    cancel = CancelToken()
    threading.Timer(1.0, lambda: cancel.cancel("test cancel")).start()
    req = RunRequest(model_alias=alias, harness="agent", profile=Profile.SMOKE, task_ids=FAST_TASKS)
    run_id = bench_env.runner().run(req, cancel)
    run_row = bench_env.store.get_run(run_id)
    assert run_row is not None and run_row["status"] == "interrupted"
    statuses = {a["status"] for a in bench_env.store.attempts_for_run(run_id)}
    assert "interrupted" in statuses
    interrupted = [
        a for a in bench_env.store.attempts_for_run(run_id) if a["status"] == "interrupted"
    ]
    assert all(a["failure_class"] == "cancelled" for a in interrupted)

    resumed = bench_env.runner().run(
        RunRequest(
            model_alias=alias,
            harness="agent",
            profile=Profile.SMOKE,
            task_ids=FAST_TASKS,
            resume_run_id=run_id,
        )
    )
    assert resumed == run_id
    final = bench_env.store.attempts_for_run(run_id)
    assert {a["status"] for a in final} == {"completed"} and all(a["passed"] == 1 for a in final)


def test_resume_refuses_changed_config(bench_env: BenchEnv) -> None:
    run_id = run(bench_env, "fake-reference", tasks=FAST_TASKS[:1])
    with pytest.raises(ConfigError, match="fingerprint"):
        run(bench_env, "fake-noop", tasks=FAST_TASKS[:1], resume_run_id=run_id)


def test_two_runs_comparable_and_isolated(bench_env: BenchEnv) -> None:
    r1 = attempts(bench_env, run(bench_env, "fake-reference"))
    r2 = attempts(bench_env, run(bench_env, "fake-reference"))
    for task_id in FAST_TASKS:
        a, b = r1[task_id], r2[task_id]
        assert a["score"] == b["score"] and a["passed"] == b["passed"]
        assert a["final_hash"] == b["final_hash"] and a["task_hash"] == b["task_hash"]
        assert a["artifact_dir"] != b["artifact_dir"]
        diff_a = bench_env.artifacts.root / a["artifact_dir"] / "diff.patch"
        diff_b = bench_env.artifacts.root / b["artifact_dir"] / "diff.patch"
        assert diff_a.read_text() == diff_b.read_text() and diff_a != diff_b
    run1 = bench_env.store.get_run(r1[FAST_TASKS[0]]["run_id"])
    run2 = bench_env.store.get_run(r2[FAST_TASKS[0]]["run_id"])
    assert run1 and run2 and run1["config_fingerprint"] == run2["config_fingerprint"]


def test_workspace_does_not_leak_between_attempts(bench_env: BenchEnv) -> None:
    run(bench_env, "fake-reference", tasks=FAST_TASKS[:1])
    a = attempts(bench_env, run(bench_env, "fake-noop", tasks=FAST_TASKS[:1]))[FAST_TASKS[0]]
    assert a["baseline_hash"] == a["final_hash"]  # noop sees the pristine fixture


def test_concurrency_and_repeats(bench_env: BenchEnv) -> None:
    run_id = run(bench_env, "fake-reference", concurrency=2, repeats=2)
    rows = bench_env.store.attempts_for_run(run_id)
    assert len(rows) == 4 and all(a["passed"] == 1 for a in rows)
    assert {a["repeat_index"] for a in rows} == {0, 1}


def test_remote_provider_requires_consent(bench_env: BenchEnv) -> None:
    with pytest.raises(ConfigError, match="REMOTE"):
        run(bench_env, "openai-paid")


def test_unknown_task_and_model(bench_env: BenchEnv) -> None:
    with pytest.raises(ConfigError, match="unknown task"):
        run(bench_env, "fake-reference", tasks=["python.nope"])
    with pytest.raises(ConfigError, match="unknown model alias"):
        run(bench_env, "no-such-model")


def test_unavailable_runtime_is_skipped_not_passed(
    bench_env: BenchEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_which = reg.shutil.which
    monkeypatch.setattr(
        reg.shutil, "which", lambda name: None if name == "g++" else real_which(name)
    )
    rows = attempts(
        bench_env, run(bench_env, "fake-reference", tasks=["cpp_sensor_fusion.kalman-cv"])
    )
    a = rows["cpp_sensor_fusion.kalman-cv"]
    assert a["status"] == "skipped" and a["passed"] is None and a["score"] is None
    assert "g++" in a["skip_reason"]


FAKE_AGENT = """
import pathlib, re, shutil, sys
prompt = sys.stdin.read()
task_id = re.search(r"# Task (\\S+) ", prompt).group(1)
for task_yaml in pathlib.Path(sys.argv[1]).rglob("task.yaml"):
    if f"id: {task_id}\\n" in task_yaml.read_text():
        ref = task_yaml.parent / "reference"
        for f in ref.rglob("*"):
            if f.is_file():
                dst = pathlib.Path(f.relative_to(ref))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
print("Copied the reference solution. Tested: no. Limitations: scripted external agent.")
"""


def test_external_cli_harness(bench_env: BenchEnv, tmp_path: Path) -> None:
    script = tmp_path / "fake_agent.py"
    script.write_text(FAKE_AGENT)
    harness = HarnessConfig(
        name="fake-cli",
        kind="cli",
        argv=[sys.executable, str(script), str(REPO / "tasks")],
        prompt_via="stdin",
    )
    bench_env.models = bench_env.models.model_copy(update={"harnesses": [harness]})
    with pytest.raises(ConfigError, match="allow-external-harness"):
        run(bench_env, "scripted-cli", harness="fake-cli")
    rows = attempts(
        bench_env, run(bench_env, "scripted-cli", harness="fake-cli", allow_external_harness=True)
    )
    for a in rows.values():
        assert a["passed"] == 1, a
        assert a["input_tokens"] is None and a["usage_reliable"] == 0  # unknown, not guessed
    run_row = bench_env.store.get_run(rows[FAST_TASKS[0]]["run_id"])
    assert run_row and run_row["provider"] == "external"
