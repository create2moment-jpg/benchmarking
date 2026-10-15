"""Performance capture through the CLI: BYOK sessions with a model endpoint, `perf probe`,
and `run --perf-probe`, all against in-process fake model servers."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from engineering_llm_bench.cli.main import app
from engineering_llm_bench.redaction import Redactor
from engineering_llm_bench.reporting import ReportOptions, build_report, to_html
from engineering_llm_bench.storage.artifacts import ArtifactStore
from engineering_llm_bench.storage.db import Store
from tests.conftest import REPO
from tests.fake_model_servers import FakeOllama, FakeVllm, serve

pytestmark = pytest.mark.integration
TASK = "python.fix-interval-merge"


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = yaml.safe_load((REPO / "configs/benchmark.example.yaml").read_text())
    data.update(
        tasks_dir=str(REPO / "tasks"),
        artifacts_dir=str(tmp_path / "art"),
        database=str(tmp_path / "art/bench.db"),
    )
    (tmp_path / "bench.yaml").write_text(yaml.safe_dump(data))
    shutil.copy(REPO / "configs/models.example.yaml", tmp_path / "models.yaml")
    monkeypatch.setenv("BENCH_CONFIG", str(tmp_path / "bench.yaml"))
    monkeypatch.setenv("BENCH_MODELS", str(tmp_path / "models.yaml"))
    return tmp_path


def store(root: Path) -> Store:
    return Store(root / "art/bench.db")


def test_byok_session_records_server_performance(cli: Path) -> None:
    runner = CliRunner()
    with serve(FakeOllama) as url:
        settings = cli / "settings.json"
        settings.write_text(
            f'{{\n  // jsonc\n  "github.copilot.chat.byok.ollamaEndpoint": "{url}",\n}}'
        )
        session = cli / "session"
        res = runner.invoke(
            app,
            [
                "byok",
                "prepare",
                "--out",
                str(session),
                "--task",
                TASK,
                "--byok-config",
                str(settings),
            ],
        )
        assert res.exit_code == 0, res.output
        assert "ollama 0.12.3" in res.output
        assert url not in (session / "starting_point.md").read_text()  # not exposed to agent
        assert runner.invoke(app, ["byok", "mark-start", str(session)]).exit_code == 0
        task_dir = next((session / "tasks").iterdir())
        time.sleep(0.3)
        shutil.copytree(
            REPO / "tasks/python/fix_interval_merge/reference", task_dir, dirs_exist_ok=True
        )
        (task_dir / "RESULT.md").write_text("Status: done\nTested.\n")
        res = runner.invoke(
            app,
            [
                "byok",
                "collect",
                str(session),
                "--model-label",
                "qwen-3.8",
                "--calibration-runs",
                "1",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "generation tokens/s          40.0" in res.output
    s = store(cli)
    run = s.list_runs()[0]
    perf = json.loads(run["perf_json"])
    assert perf["endpoint"] == url and perf["model"] == "qwen3.8:latest"
    assert perf["start"]["server"]["server"] == "ollama"
    assert perf["timing"]["tasks"][0]["estimated_s"] >= 0.3
    assert perf["session_counter_delta"] is None  # Ollama exposes no counters: unknown, not 0
    report = build_report(s, ArtifactStore(cli / "art"), Redactor(), ReportOptions())
    group = report["groups"]["qwen-3.8 | copilot-byok"]
    assert (
        group["server_perf"]["gen_tps"] == 40.0 and group["server_perf"]["quantization"] == "Q4_K_M"
    )
    assert "Q4_K_M" in to_html(report)


def test_byok_prepare_rejects_ambiguous_config_before_creating(cli: Path) -> None:
    settings = cli / "s.json"
    settings.write_text(
        '{"a": {"url": "http://localhost:1/v1"}, "b": {"url": "http://localhost:2/v1"}}'
    )
    res = CliRunner().invoke(
        app,
        [
            "byok",
            "prepare",
            "--out",
            str(cli / "x"),
            "--task",
            TASK,
            "--byok-config",
            str(settings),
        ],
    )
    assert res.exit_code == 2 and "several endpoints" in res.output
    assert not (cli / "x").exists()


def test_perf_probe_attach_and_run_perf_probe(cli: Path) -> None:
    runner = CliRunner()
    with serve(FakeVllm) as url:
        models = yaml.safe_load((cli / "models.yaml").read_text())
        models["models"].append(
            {
                "alias": "local-vllm-fake",
                "provider": "openai_compatible",
                "base_url": url + "/v1",
                "model_id": "strata/qwen-3.8",
            }
        )
        (cli / "models.yaml").write_text(yaml.safe_dump(models))
        res = runner.invoke(
            app,
            [
                "run",
                "--model",
                "local-vllm-fake",
                "--harness",
                "direct",
                "--task",
                TASK,
                "--perf-probe",
                "--calibration-runs",
                "1",
            ],
        )
        assert res.exit_code == 0, res.output
        run = store(cli).list_runs()[0]
        perf = json.loads(run["perf_json"])
        assert perf["end"]["server"]["server"] == "vllm"
        assert perf["session_counter_delta"]["vllm:prompt_tokens_total"] >= 0
        attempt = store(cli).attempts_for_run(run["id"])[0]
        assert attempt["model_time_s"] is not None and attempt["eval_cpu_s"] is not None
        out = cli / "probe.json"
        res = runner.invoke(
            app,
            [
                "perf",
                "probe",
                "--endpoint",
                url,
                "--calibration-runs",
                "1",
                "--out",
                str(out),
                "--attach-run",
                run["id"],
            ],
        )
        assert res.exit_code == 0, res.output
        assert json.loads(out.read_text())["server"]["server"] == "vllm"
    res = runner.invoke(app, ["perf", "probe", "--endpoint", "https://api.example.com/v1"])
    assert res.exit_code != 0 and "allow-remote" in res.output
