"""One-command BYOK workflow: endpoint/model auto-detection and `bench byok start` / `finish`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import engineering_llm_bench.byok.autodetect as auto
from engineering_llm_bench.cli.main import app
from tests.conftest import REPO
from tests.fake_model_servers import MODEL, FakeOllama, FakeVllm, serve

pytestmark = pytest.mark.integration


def test_find_endpoint_from_settings_and_guess_model(tmp_path: Path) -> None:
    with serve(FakeOllama) as url:
        settings = tmp_path / "settings.json"
        settings.write_text(
            '{\n // comment\n "github.copilot.chat.byok.ollamaEndpoint": "' + url + '"\n}'
        )
        found = auto.find_endpoint([settings])
        assert found is not None and found.base_url == url.rstrip("/")
        assert auto.is_ollama(found.base_url)
        guess, choices = auto.guess_model(found)
        assert guess == MODEL and choices == [MODEL]  # the one model loaded right now
        assert auto.endpoint_is_local(found.base_url)


def test_guess_model_openai_compatible() -> None:
    with serve(FakeVllm) as url:
        guess, _ = auto.guess_model(auto.FoundEndpoint(url.rstrip("/") + "/v1", "test", []))
        assert guess  # single model offered by /v1/models


def test_no_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto, "OLLAMA_DEFAULT", "http://127.0.0.1:9")
    assert auto.find_endpoint([]) is None
    assert auto.guess_model(None) == (None, [])


def test_start_then_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "bench.yaml"
    data = (REPO / "configs/benchmark.example.yaml").read_text()
    data = data.replace("tasks_dir: tasks", f"tasks_dir: {REPO / 'tasks'}")
    data = data.replace("artifacts_dir: artifacts", f"artifacts_dir: {tmp_path / 'art'}")
    data = data.replace("database: artifacts/bench.db", f"database: {tmp_path / 'art/bench.db'}")
    data = data.replace("outputs_dir: artifacts/model_outputs", f"outputs_dir: {tmp_path / 'mo'}")
    cfg.write_text(data)
    monkeypatch.setenv("BENCH_CONFIG", str(cfg))
    monkeypatch.setenv("BENCH_MODELS", str(REPO / "configs/models.example.yaml"))
    monkeypatch.setattr(auto, "copy_to_clipboard", lambda text: True)
    cli = CliRunner()
    session = tmp_path / "sessions" / "s1"
    runtime = tmp_path / "sessions" / ".runtime-venv"
    with serve(FakeOllama) as url:
        monkeypatch.setattr(
            auto, "find_endpoint", lambda: auto.FoundEndpoint(url.rstrip("/"), "test", [])
        )
        res = cli.invoke(
            app,
            ["byok", "start", "--task", "python.fix-interval-merge", "--out", str(session),
             "--python-env", str(runtime), "--no-open", "--auto-approve"],
        )  # fmt: skip
        assert res.exit_code == 0, res.output
        assert "on the clipboard" in res.output and "finish.sh" in res.output
        settings = json.loads((session / ".vscode/settings.json").read_text())
        assert settings["python.defaultInterpreterPath"].startswith(str(runtime))
        approve = settings["chat.tools.terminal.autoApprove"]
        assert approve["pytest"] is True and approve["g++"] is True
        assert approve["rm"] is False and approve["git"] is False and approve["pip"] is False
        edits = settings["chat.tools.edits.autoApprove"]
        assert edits["**/*"] is True and edits["**/TASK.md"] is False
        assert str(runtime) in (session / "AGENTS.md").read_text()
        assert str(runtime) in (session / "starting_point.md").read_text()
        # A second start is refused until this one is collected.
        res = cli.invoke(
            app, ["byok", "start", "--out", str(tmp_path / "sessions/s2"), "--no-open"]
        )
        assert res.exit_code != 0 and "not been collected" in res.output

        task = next((session / "tasks").iterdir())
        shutil.copytree(
            REPO / "tasks/python/fix_interval_merge/reference", task, dirs_exist_ok=True
        )
        (task / "RESULT.md").write_text("Status: done\n")
        res = cli.invoke(app, ["byok", "finish", "--no-serve", "--calibration-runs", "0"])
    assert res.exit_code == 0, res.output
    assert f"model: {MODEL}" in res.output and "passed 1/1" in res.output
    assert not (session / "tasks").exists()
    assert not (tmp_path / "art" / auto.CURRENT_SESSION_FILE).exists()
    res = cli.invoke(app, ["byok", "finish", "--no-serve"])
    assert res.exit_code != 0 and "no session to collect" in res.output


def test_no_auto_approve_by_default(tmp_path: Path) -> None:
    from engineering_llm_bench.byok.session import vscode_settings  # noqa: PLC0415

    assert vscode_settings(None, False) == {}
    assert "chat.tools.terminal.autoApprove" not in vscode_settings(tmp_path, False)
