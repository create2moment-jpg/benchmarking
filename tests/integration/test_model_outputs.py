"""Per-model output folders and their invisibility to other models' commands."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from engineering_llm_bench.byok import collect_session, prepare_session
from engineering_llm_bench.cli.main import app
from engineering_llm_bench.config.models import HarnessConfig
from engineering_llm_bench.domain import Profile
from engineering_llm_bench.orchestration.runner import RunRequest
from engineering_llm_bench.outputs import ModelOutputs, model_slug
from engineering_llm_bench.tools.runner import mount_hiding_available
from tests.conftest import FAST_TASKS, REPO, BenchEnv

pytestmark = pytest.mark.integration
needs_root_bypass = pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores permission bits; lock not observable"
)


def outputs_of(env: BenchEnv) -> ModelOutputs:
    return ModelOutputs(env.artifacts.root / "model_outputs")


def run(env: BenchEnv, alias: str, harness: str = "agent", **kw: Any) -> str:
    req = RunRequest(
        model_alias=alias, harness=harness, profile=Profile.SMOKE, task_ids=FAST_TASKS, **kw
    )
    return env.runner().run(req)


# ---------------------------------------------------------------------------- ModelOutputs
def test_model_slug() -> None:
    assert model_slug("strata/qwen-3.8") == "strata-qwen-3.8"
    assert model_slug("../..") == "model"
    assert model_slug("a b:c") == "a-b-c"


def test_save_lock_list_export(tmp_path: Path) -> None:
    out = ModelOutputs(tmp_path / "mo")
    rel = out.save_attempt(
        "strata/qwen-3.8",
        "RUN1",
        "python.x",
        0,
        {"pkg/a.py": b"x = 1\n", "../escape.py": b"bad"},
        "--- a\n+++ b\n",
        {"passed": True},
        {"provider": "ollama"},
    )
    assert rel == Path("strata-qwen-3.8/runs/RUN1/python.x__r0")
    assert out.locked and stat.S_IMODE(out.root.stat().st_mode) == 0
    with out.unlocked() as root:
        target = root / rel
        assert (target / "code/pkg/a.py").read_bytes() == b"x = 1\n"
        assert not (target / "escape.py").exists() and not (root / "escape.py").exists()
        assert json.loads((target / "meta.json").read_text())["passed"] is True
        info = json.loads((root / "strata-qwen-3.8/MODEL.json").read_text())
        assert info["alias"] == "strata/qwen-3.8" and info["provider"] == "ollama"
    assert out.locked  # re-locked after use
    [row] = out.list_models()
    assert row["alias"] == "strata/qwen-3.8" and row["attempts"] == 1
    exported = out.export("strata/qwen-3.8", tmp_path / "export")
    assert (exported / "runs/RUN1/python.x__r0/code/pkg/a.py").is_file()
    with pytest.raises(FileNotFoundError):
        out.export("other", tmp_path / "export")


def test_lock_disabled(tmp_path: Path) -> None:
    out = ModelOutputs(tmp_path / "mo", lock=False)
    out.save_attempt("m", "R", "t", 0, {"a": b"1"}, "", {})
    assert not out.locked


@needs_root_bypass
def test_locked_folder_denies_access(tmp_path: Path) -> None:
    out = ModelOutputs(tmp_path / "mo")
    out.save_attempt("m", "R", "t", 0, {"a": b"1"}, "", {})
    with pytest.raises(PermissionError):
        list(out.root.iterdir())


# ---------------------------------------------------------------------------- runs
def test_each_model_gets_its_own_folder(bench_env: BenchEnv) -> None:
    ref_run = run(bench_env, "fake-reference")
    noop_run = run(bench_env, "fake-noop")
    out = outputs_of(bench_env)
    assert out.locked or os.name == "nt"
    with out.unlocked() as root:
        assert sorted(p.name for p in root.iterdir()) == ["fake-noop", "fake-reference"]
        for alias, run_id in (("fake-reference", ref_run), ("fake-noop", noop_run)):
            for task_id in FAST_TASKS:
                d = root / alias / "runs" / run_id / f"{task_id}__r0"
                assert (d / "code").is_dir() and (d / "diff.patch").is_file()
                meta = json.loads((d / "meta.json").read_text())
                assert meta["task_id"] == task_id and meta["model_alias"] == alias
        ref_diff = root / "fake-reference/runs" / ref_run / f"{FAST_TASKS[0]}__r0/diff.patch"
        noop_diff = root / "fake-noop/runs" / noop_run / f"{FAST_TASKS[0]}__r0/diff.patch"
        assert ref_diff.read_text() and not noop_diff.read_text()
    for a in bench_env.store.attempts_for_run(ref_run):
        names = {r["name"] for r in bench_env.store.artifacts(a["id"])}
        assert "code_location.txt" in names


PROBE_AGENT = """
import json, os, pathlib, sys
result = {}
for d in sys.argv[1:]:
    try:
        result[d] = sorted(os.listdir(d))
    except OSError as e:
        result[d] = f"error {e.errno}"
pathlib.Path("probe.json").write_text(json.dumps(result))
print("Probed. Tested: no. Limitations: probe.")
"""


@pytest.mark.skipif(not mount_hiding_available(), reason="no mount namespaces here")
def test_external_agent_cannot_see_other_models_or_tasks(
    bench_env: BenchEnv, tmp_path: Path
) -> None:
    run(bench_env, "fake-reference")  # another model's code now exists
    script = tmp_path / "probe_agent.py"
    script.write_text(PROBE_AGENT)
    outputs_root = bench_env.artifacts.root / "model_outputs"
    targets = [str(outputs_root), str(bench_env.artifacts.root), str(REPO / "tasks")]
    harness = HarnessConfig(
        name="probe-cli",
        kind="cli",
        argv=[sys.executable, str(script), *targets],
        prompt_via="stdin",
    )
    bench_env.models = bench_env.models.model_copy(update={"harnesses": [harness]})
    run_id = run(bench_env, "scripted-cli", harness="probe-cli", allow_external_harness=True)
    out = outputs_of(bench_env)
    with out.unlocked() as root:
        probe = root / "scripted-cli/runs" / run_id / f"{FAST_TASKS[0]}__r0/code/probe.json"
        seen = json.loads(probe.read_text())
    # Empty, or absent altogether when it lies inside another hidden folder (ENOENT).
    assert all(v in ([], "error 2") for v in seen.values()), seen


# ---------------------------------------------------------------------------- BYOK
def test_byok_collect_moves_code_into_model_folder(bench_env: BenchEnv, tmp_path: Path) -> None:
    session = tmp_path / "session"
    prepare_session(
        bench_env.registry,
        bench_env.bench,
        bench_env.artifacts.root,
        session,
        Profile.SMOKE,
        [FAST_TASKS[0]],
    )
    task_dir = next((session / "tasks").iterdir())
    (task_dir / "RESULT.md").write_text("Status: done\n")

    def collect(**kw: Any):  # type: ignore[no-untyped-def]
        return collect_session(
            session,
            bench_env.registry,
            bench_env.bench,
            bench_env.store,
            bench_env.artifacts,
            bench_env.artifacts.root,
            model_label="strata/qwen-3.8",
            **kw,
        )

    kept = collect(keep_session=True)
    assert not kept.session_removed and (session / "tasks").is_dir()
    result = collect(force=True)
    assert result.session_removed
    assert not (session / "tasks").exists() and not (session / "starting_point.md").exists()
    assert (session / "COLLECTED.md").is_file()
    out = outputs_of(bench_env)
    with out.unlocked() as root:
        archive = root / "strata-qwen-3.8/byok_sessions" / result.session_id
        assert result.archived_to == archive
        assert (archive / "tasks" / task_dir.name / "RESULT.md").is_file()
        runs = sorted(p.name for p in (root / "strata-qwen-3.8/runs").iterdir())
        assert runs == sorted([kept.run_id, result.run_id])
    # The working copy is gone, but a forced re-collection still works from the archive.
    again = collect(force=True)
    assert again.run_id not in (kept.run_id, result.run_id)
    assert {a["task_id"] for a in bench_env.store.attempts_for_run(again.run_id)} == {FAST_TASKS[0]}


# ---------------------------------------------------------------------------- CLI
def test_outputs_cli(bench_env: BenchEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "bench.yaml"
    data = (REPO / "configs/benchmark.example.yaml").read_text()
    data = data.replace("tasks_dir: tasks", f"tasks_dir: {REPO / 'tasks'}")
    data = data.replace("artifacts_dir: artifacts", f"artifacts_dir: {tmp_path / 'art'}")
    data = data.replace("database: artifacts/bench.db", f"database: {tmp_path / 'art/bench.db'}")
    data = data.replace(
        "outputs_dir: artifacts/model_outputs", f"outputs_dir: {tmp_path / 'art/mo'}"
    )
    cfg.write_text(data)
    monkeypatch.setenv("BENCH_CONFIG", str(cfg))
    monkeypatch.setenv("BENCH_MODELS", str(REPO / "configs/models.example.yaml"))
    cli = CliRunner()
    res = cli.invoke(
        app, ["run", "--model", "fake-reference", "--task", FAST_TASKS[0], "--profile", "smoke"]
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "art/mo").is_dir()
    res = cli.invoke(app, ["outputs", "list"])
    assert res.exit_code == 0 and "fake-reference" in res.output, res.output
    res = cli.invoke(app, ["outputs", "path", "--model", "fake-reference"])
    assert res.exit_code == 0 and "fake-reference" in res.output, res.output
    res = cli.invoke(
        app, ["outputs", "export", "--model", "fake-reference", "--to", str(tmp_path / "x")]
    )
    assert res.exit_code == 0, res.output
    assert any((tmp_path / "x/fake-reference/runs").iterdir())
    assert cli.invoke(app, ["outputs", "unlock"]).exit_code == 0
    assert stat.S_IMODE((tmp_path / "art/mo").stat().st_mode) == 0o700
    assert cli.invoke(app, ["outputs", "lock"]).exit_code == 0
    assert stat.S_IMODE((tmp_path / "art/mo").stat().st_mode) == 0
