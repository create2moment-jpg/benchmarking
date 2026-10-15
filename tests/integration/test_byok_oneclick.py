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


def test_container_session(tmp_path: Path) -> None:
    from engineering_llm_bench.byok.session import ContainerSpec, prepare_session  # noqa: PLC0415
    from engineering_llm_bench.catalog.registry import TaskRegistry  # noqa: PLC0415
    from engineering_llm_bench.config.loader import load_benchmark  # noqa: PLC0415
    from engineering_llm_bench.domain import Profile  # noqa: PLC0415

    bench = load_benchmark(REPO / "configs/benchmark.example.yaml")
    session = tmp_path / "s"
    prepare_session(
        TaskRegistry.discover(REPO / "tasks"),
        bench,
        tmp_path / "art",
        session,
        Profile.SMOKE,
        ["python.fix-interval-merge"],
        python_env=tmp_path / "venv",  # ignored in container mode
        container=ContainerSpec("elb-tasks:abc"),
    )
    dc = json.loads((session / ".devcontainer/devcontainer.json").read_text())
    assert dc["image"] == "elb-tasks:abc" and dc["workspaceFolder"] == "/workspace"
    assert "--network=host" in dc["runArgs"] and dc["remoteUser"] == "dev"
    settings = dc["customizations"]["vscode"]["settings"]
    assert settings["chat.tools.terminal.autoApprove"] == {"/.*/": True}
    assert settings["chat.tools.edits.autoApprove"]["**/TASK.md"] is False
    # Nothing is auto-approved if the folder is opened on the host instead of the container.
    assert not (session / ".vscode/settings.json").exists()
    start = (session / "starting_point.md").read_text()
    assert "disposable container" in start and "Do not install packages" not in start
    assert "disposable container" in (session / "AGENTS.md").read_text()
    assert str(tmp_path / "venv") not in start


def test_devcontainer_uri() -> None:
    uri = auto.devcontainer_uri(Path("/home/u/s"))
    assert uri == "vscode-remote://dev-container+" + b"/home/u/s".hex() + "/workspace"


def test_docker_requirements_match_pyproject() -> None:
    import tomllib  # noqa: PLC0415

    def reqs(path: Path) -> list[str]:
        lines = path.read_text().splitlines()
        return sorted(x.strip() for x in lines if x.strip() and not x.startswith("#"))

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert reqs(REPO / "docker/requirements-bench.txt") == sorted(project["dependencies"])
    tasks = reqs(REPO / "docker/requirements-tasks.txt")
    names = {r.split(">")[0].split("=")[0].lower() for r in tasks}
    assert {"pytest", "ruff", "numpy", "duckdb"} <= names


def test_server_profile_start_to_finish(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os  # noqa: PLC0415
    import stat  # noqa: PLC0415
    import time  # noqa: PLC0415

    from engineering_llm_bench.dashboard.server import DashboardApi  # noqa: PLC0415
    from engineering_llm_bench.redaction import Redactor  # noqa: PLC0415
    from engineering_llm_bench.storage.artifacts import ArtifactStore  # noqa: PLC0415
    from engineering_llm_bench.storage.db import Store  # noqa: PLC0415
    from tests.unit.test_remote_profiling import FAKE_SMI  # noqa: PLC0415

    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
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
    monkeypatch.setattr(auto, "find_endpoint", lambda: None)
    cli = CliRunner()
    session = tmp_path / "sessions" / "s1"
    log = tmp_path / "ollama.log"
    with serve(FakeOllama) as url:
        server_yaml = tmp_path / "server.yaml"
        server_yaml.write_text(
            f"server:\n  kind: ollama\n  url: {url}\n  log_command: cat {log}\n"
            "monitor:\n  interval_s: 1\n  extra_commands: ['echo strata-status-ok']\n"
        )
        res = cli.invoke(
            app,
            ["byok", "start", "--task", "python.fix-interval-merge", "--out", str(session),
             "--no-open", "--server-config", str(server_yaml)],
        )  # fmt: skip
        assert res.exit_code == 0, res.output
        assert "monitoring ollama at" in res.output
        task = next((session / "tasks").iterdir())
        shutil.copytree(
            REPO / "tasks/python/fix_interval_merge/reference", task, dirs_exist_ok=True
        )
        (task / "RESULT.md").write_text("Status: done\n")
        stamp = time.strftime("%Y/%m/%d - %H:%M:%S", time.localtime())
        offset = time.strftime("%z")
        log.write_text(
            f'[GIN] {stamp} | 200 |  3.5s |  127.0.0.1 | POST     "/api/chat"\n'
            f'[GIN] {stamp} | 200 |  1.5s |  127.0.0.1 | POST     "/api/chat"\n'
        )
        assert offset  # the snapshot records the server's UTC offset the same way
        time.sleep(3.5)  # a few monitor samples
        res = cli.invoke(app, ["byok", "finish", "--no-serve", "--calibration-runs", "0"])
    assert res.exit_code == 0, res.output
    assert "time profile:" in res.output and "LLM requests         2" in res.output
    store = Store(tmp_path / "art/bench.db")
    run = store.list_runs()[0]
    profile = json.loads(run["perf_json"])["profile"]
    assert profile["llm_requests"] == 2 and profile["model_time_s"] == 5.0
    assert profile["gpu_util_peak_pct"] == 87.0 and profile["gpu_mem_total_mb"] == 24564
    assert profile["samples"] >= 2 and profile["loaded_models"] == [MODEL]
    assert profile["agent_duration_s"] is not None and profile["setup_to_finish_s"] > 3
    assert profile["tasks"][0]["task_id"] == "python.fix-interval-merge"
    # The monitor process has stopped.
    manifest_remote = json.loads(next((tmp_path / "art/byok_sessions").glob("*.json")).read_text())[
        "remote"
    ]
    assert "strata-status-ok" in manifest_remote["end"]["extra"]["CMD0"]
    pid = manifest_remote["monitor_pid"]
    time.sleep(0.5)
    assert (
        not Path(f"/proc/{pid}").exists()
        or b"byok" not in Path(f"/proc/{pid}/cmdline").read_bytes()
    )
    # The dashboard payload carries the profile for the overview / time-profile section.
    api = DashboardApi(store, ArtifactStore(tmp_path / "art", Redactor()), Redactor())
    report_run = api.report()["runs"][0]
    assert report_run["performance"]["profile"]["llm_requests"] == 2
    store.close()
