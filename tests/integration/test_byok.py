"""Copilot BYOK workflow: prepare a session for an IDE agent, then collect and score its output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engineering_llm_bench.byok import collect_session, prepare_session
from engineering_llm_bench.cli.main import app
from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.domain import Profile
from tests.conftest import REPO, BenchEnv

pytestmark = pytest.mark.integration
TASKS = ["python.fix-interval-merge", "sql.latest-sensor-reading", "debugging.timestamp-sync"]


def prepare(env: BenchEnv, tmp_path: Path) -> Path:
    out = tmp_path / "session"
    prepare_session(env.registry, env.bench, env.artifacts.root, out, Profile.SMOKE, TASKS)
    return out


def folder(session: Path, task_id: str) -> Path:
    return next(p for p in (session / "tasks").iterdir() if p.name.endswith(task_id))


def collect(env: BenchEnv, session: Path, **kw: object):  # type: ignore[no-untyped-def]
    return collect_session(
        session,
        env.registry,
        env.bench,
        env.store,
        env.artifacts,
        env.artifacts.root,
        model_label="strata/qwen-3.8",
        **kw,
    )  # type: ignore[arg-type]


def test_session_contents_hide_evaluators_and_model(bench_env: BenchEnv, tmp_path: Path) -> None:
    session = prepare(bench_env, tmp_path)
    start = (session / "starting_point.md").read_text()
    for task_id in TASKS:
        assert task_id in start
        assert (folder(session, task_id) / "TASK.md").is_file()
    lowered = start.lower()
    for word in ("ollama", "qwen", "copilot", "model label"):
        assert word not in lowered
    all_text = "\n".join(p.read_text(errors="ignore") for p in session.rglob("*") if p.is_file())
    assert "test_hidden" not in all_text
    reference = (REPO / "tasks/python/fix_interval_merge/reference/intervals.py").read_text()
    assert reference not in all_text
    for rel in ("AGENTS.md", ".github/copilot-instructions.md"):
        assert "starting_point.md" in (session / rel).read_text()
    manifest = json.loads((session / ".bench/session.json").read_text())
    assert [t["task_id"] for t in manifest["tasks"]] == sorted(TASKS)


def test_prepare_refuses_repo_and_non_empty(bench_env: BenchEnv, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="inside the benchmark repository"):
        prepare_session(
            bench_env.registry,
            bench_env.bench,
            bench_env.artifacts.root,
            REPO / "byok_sessions" / "x",
            Profile.SMOKE,
            TASKS,
        )
    session = prepare(bench_env, tmp_path)
    with pytest.raises(ConfigError, match="not empty"):
        prepare_session(
            bench_env.registry,
            bench_env.bench,
            bench_env.artifacts.root,
            session,
            Profile.SMOKE,
            TASKS,
        )


def test_collect_scores_agent_output(bench_env: BenchEnv, tmp_path: Path) -> None:
    session = prepare(bench_env, tmp_path)
    # Task 1: correct solution + RESULT.md.
    py = folder(session, "python.fix-interval-merge")
    shutil.copytree(REPO / "tasks/python/fix_interval_merge/reference", py, dirs_exist_ok=True)
    (py / "RESULT.md").write_text("Status: done\nFixed. Verified with python -m pytest -q.\n")
    # Task 2: wrong answer that *claims* success.
    sql = folder(session, "sql.latest-sensor-reading")
    shutil.copy(REPO / "tasks/sql/latest_sensor_reading/broken/solution.sql", sql / "solution.sql")
    (sql / "RESULT.md").write_text("Status: done\nAll tests pass.\n")
    # Task 3: cheat attempt - a non-editable conftest.py that skips every test.
    dbg = folder(session, "debugging.timestamp-sync")
    (dbg / "conftest.py").write_text(
        "import pytest\n\ndef pytest_collection_modifyitems(items):\n"
        "    for i in items:\n        i.add_marker(pytest.mark.skip)\n"
    )
    (dbg / "TASK.md").write_text("changed")

    dry = collect(bench_env, session, dry_run=True)
    assert dry.run_id is None and bench_env.store.list_runs() == []
    subs = {s.task_id: s for s in dry.submissions}
    assert subs["debugging.timestamp-sync"].rejected == ["conftest.py"]
    assert any("TASK.md was modified" in n for n in subs["debugging.timestamp-sync"].notes)

    result = collect(bench_env, session, provider_label="ollama", remote=False, session_minutes=12)
    attempts = {a["task_id"]: a for a in bench_env.store.attempts_for_run(result.run_id)}
    assert attempts["python.fix-interval-merge"]["passed"] == 1
    assert attempts["sql.latest-sensor-reading"]["passed"] == 0  # self-report not trusted
    assert attempts["debugging.timestamp-sync"]["passed"] == 0  # cheat not applied
    run = bench_env.store.get_run(result.run_id)
    assert run and run["model_alias"] == "strata/qwen-3.8" and run["harness"] == "copilot-byok"
    assert run["mode"] == "manual" and run["provider"] == "ollama" and run["remote"] == 0
    assert "12" in run["notes"]
    notes = json.loads(attempts["sql.latest-sensor-reading"]["notes_json"])
    assert any("self-reported status: done" in n for n in notes)
    assert attempts["python.fix-interval-merge"]["summary"].startswith("Status: done")

    with pytest.raises(ConfigError, match="already collected"):
        collect(bench_env, session)
    again = collect(bench_env, session, force=True)
    assert again.run_id != result.run_id


def test_only_attempted_skips_untouched(bench_env: BenchEnv, tmp_path: Path) -> None:
    session = prepare(bench_env, tmp_path)
    result = collect(bench_env, session, only_attempted=True)
    statuses = {a["status"] for a in bench_env.store.attempts_for_run(result.run_id)}
    assert statuses == {"skipped"}


def test_tampered_manifest_rejected(bench_env: BenchEnv, tmp_path: Path) -> None:
    session = prepare(bench_env, tmp_path)
    path = session / ".bench/session.json"
    data = json.loads(path.read_text())
    data["tasks"] = data["tasks"][:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="trusted copy"):
        collect(bench_env, session)


def test_byok_scripts_via_cli(
    bench_env: BenchEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "bench.yaml"
    data = (REPO / "configs/benchmark.example.yaml").read_text()
    data = data.replace("tasks_dir: tasks", f"tasks_dir: {REPO / 'tasks'}")
    data = data.replace("artifacts_dir: artifacts", f"artifacts_dir: {tmp_path / 'art'}")
    data = data.replace("database: artifacts/bench.db", f"database: {tmp_path / 'art/bench.db'}")
    cfg.write_text(data)
    monkeypatch.setenv("BENCH_CONFIG", str(cfg))
    monkeypatch.setenv("BENCH_MODELS", str(REPO / "configs/models.example.yaml"))
    runner = CliRunner()
    out = tmp_path / "s"
    res = runner.invoke(app, ["byok", "prepare", "--out", str(out), "--task", TASKS[0]])
    assert res.exit_code == 0, res.output
    assert "starting_point.md" in res.output
    assert "NO SESSION FOLDER OPEN" in res.output  # the prompt to send is printed
    res = runner.invoke(app, ["byok", "collect", str(out), "--model-label", "qwen-3.8"])
    assert res.exit_code == 0, res.output
    assert "passed 0/1" in res.output


def test_agent_prompt_matches_instructions_folder() -> None:
    from engineering_llm_bench.byok import agent_prompt  # noqa: PLC0415

    packaged = (REPO / "src/engineering_llm_bench/byok/command.md").read_text()
    assert (REPO / "instructions/command.md").read_text() == packaged
    prompt = agent_prompt()
    assert "starting_point.md" in prompt
    assert "NO SESSION FOLDER OPEN" in prompt  # never search the machine for a session
    assert "\n" not in prompt
    for word in ("ollama", "qwen", "vllm"):
        assert word not in prompt.lower()
