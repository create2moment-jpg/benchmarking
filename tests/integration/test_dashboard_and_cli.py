"""Acceptance 7 (dashboard) and CLI behaviour (help, exit codes, no hidden side effects)."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from engineering_llm_bench.cli.main import app
from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.dashboard.server import DashboardApi, make_handler
from engineering_llm_bench.domain import Dimension, Profile
from engineering_llm_bench.evaluation.review import (
    export_blind_packet,
    import_reviews,
    judge_attempts,
)
from engineering_llm_bench.orchestration.runner import RunRequest
from engineering_llm_bench.providers import create_provider
from engineering_llm_bench.reporting.stats import review_summary
from tests.conftest import FAST_TASKS, REPO, BenchEnv

pytestmark = pytest.mark.integration


@pytest.fixture
def server(bench_env: BenchEnv) -> Iterator[str]:
    api = DashboardApi(bench_env.store, bench_env.artifacts, bench_env.artifacts.redactor)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(api))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - local test server
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_dashboard_empty_state(server: str) -> None:
    status, html = get(server + "/")
    assert status == 200 and "No benchmark runs yet" in html
    status, body = get(server + "/api/report")
    assert status == 200 and json.loads(body)["runs"] == []


def test_dashboard_serves_persisted_runs(bench_env: BenchEnv, server: str) -> None:
    runner = bench_env.runner()
    ok = runner.run(
        RunRequest(
            model_alias="fake-reference",
            harness="agent",
            profile=Profile.SMOKE,
            task_ids=FAST_TASKS,
        )
    )
    bench_env.models = bench_env.models.model_copy(
        update={
            "models": [
                *bench_env.models.models,
                ModelConfig.model_validate(
                    {
                        "alias": "fake-error",
                        "provider": "fake",
                        "model_id": "x",
                        "fake": {"behavior": "error"},
                    }
                ),
            ]
        }
    )
    failed = bench_env.runner().run(
        RunRequest(
            model_alias="fake-error",
            harness="agent",
            profile=Profile.SMOKE,
            task_ids=FAST_TASKS[:1],
        )
    )
    report = json.loads(get(server + "/api/report")[1])
    assert {r["id"] for r in report["runs"]} == {ok, failed}
    assert "fake-reference | agent" in report["groups"] and "fake-error | agent" in report["groups"]
    failing = next(a for a in report["attempts"] if a["run_id"] == failed)
    assert failing["failure_class"] == "provider_error"
    filtered = json.loads(get(server + f"/api/report?run_ids={ok}")[1])
    assert {a["run_id"] for a in filtered["attempts"]} == {ok}
    detail = json.loads(get(server + f"/api/attempt/{failing['id']}")[1])
    assert detail["attempt"]["input_tokens"] is not None or detail["attempt"]["usage_reliable"] in (
        0,
        1,
    )
    assert detail["environment"]["bench_version"]
    names = {a["name"] for a in detail["artifacts"]}
    assert "trace.jsonl" in names
    status, text = get(server + f"/api/artifact?attempt={failing['id']}&name=trace.jsonl")
    assert status == 200 and "fake provider configured to fail" in text
    assert get(server + f"/api/artifact?attempt={failing['id']}&name=../../bench.db")[0] == 404
    assert get(server + "/api/attempt/does-not-exist")[0] == 404
    for asset in ("/app.js", "/style.css"):
        assert get(server + asset)[0] == 200


def test_dashboard_in_browser(bench_env: BenchEnv, server: str) -> None:
    """Renders the real page in headless Chromium when Playwright is installed."""
    sync_api = pytest.importorskip("playwright.sync_api")
    for alias in ("fake-reference", "fake-noop"):
        bench_env.runner().run(
            RunRequest(
                model_alias=alias, harness="agent", profile=Profile.SMOKE, task_ids=FAST_TASKS
            )
        )
    chromium = Path("/opt/pw-browsers/chromium")
    with sync_api.sync_playwright() as p:
        kwargs = {"executable_path": str(chromium)} if chromium.exists() else {}
        try:
            browser = p.chromium.launch(**kwargs)
        except Exception as exc:  # browser binary unavailable
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server + "/")
        page.wait_for_selector("#tiles .tile")
        assert "Pass rate" in page.inner_text("#tiles")
        page.click("button[data-tab=compare]")
        assert "fake-noop | agent" in page.inner_text("#compare-table")
        page.click("#group-picker label:has-text('fake-noop') input")  # deselect a model
        assert "fake-noop | agent" not in page.inner_text("#compare-table")
        page.click("button[data-tab=history]")
        page.select_option("#f-task", "sql.latest-sensor-reading")
        assert "2 attempt(s)" in page.inner_text("#history-count")
        page.click("#history-table tr.click >> nth=0")
        page.wait_for_selector("#detail h2")
        assert "Score breakdown" in page.inner_text("#detail")
        browser.close()
        assert errors == []


def test_blind_review_and_judge(bench_env: BenchEnv, tmp_path: Path) -> None:
    run_id = bench_env.runner().run(
        RunRequest(
            model_alias="fake-reference",
            harness="agent",
            profile=Profile.SMOKE,
            task_ids=FAST_TASKS,
        )
    )
    packet, key = export_blind_packet(
        bench_env.store,
        bench_env.artifacts,
        bench_env.registry,
        [run_id],
        tmp_path / "rev",
        [Dimension.EXPLANATION],
    )
    text = packet.read_text()
    assert "fake-reference" not in text and run_id not in text
    blind_ids = [i["blind_id"] for i in json.loads(text)["items"]]
    csv_file = tmp_path / "reviews.csv"
    rows = ["blind_id,reviewer,dimension,score,rationale"]
    rows += [f"{b},alice,explanation,0.9,clear" for b in blind_ids]
    rows += [f"{b},bob,explanation,0.4,terse" for b in blind_ids]
    rows += ["unknown,carol,explanation,0.5,x", f"{blind_ids[0]},dave,explanation,7,x"]
    csv_file.write_text("\n".join(rows) + "\n")
    result = import_reviews(bench_env.store, key, csv_file)
    assert result.recorded == 2 * len(blind_ids) and len(result.errors) == 2
    summary = review_summary(bench_env.store.reviews())
    assert all(i["disagreement"] == pytest.approx(0.5) for i in summary["items"])
    assert summary["high_disagreement"]
    # A judge whose replies are not valid JSON is recorded as errors, never as scores.
    judge = create_provider(bench_env.models.model("fake-reference"))
    judged = judge_attempts(
        bench_env.store,
        bench_env.artifacts,
        bench_env.registry,
        judge,
        run_id,
        Dimension.EXPLANATION,
    )
    assert judged.recorded == 0 and judged.errors


# ------------------------------------------------------------------------------------------- CLI
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = yaml.safe_load((REPO / "configs/benchmark.example.yaml").read_text())
    data.update(
        tasks_dir=str(REPO / "tasks"),
        artifacts_dir=str(tmp_path / "artifacts"),
        database=str(tmp_path / "artifacts/bench.db"),
    )
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(yaml.safe_dump(data))
    shutil.copy(REPO / "configs/models.example.yaml", tmp_path / "models.yaml")
    monkeypatch.setenv("BENCH_CONFIG", str(cfg))
    monkeypatch.setenv("BENCH_MODELS", str(tmp_path / "models.yaml"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_help_and_validate(cli_env: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["validate-config"])
    assert result.exit_code == 0, result.output
    assert "14 tasks" in result.output or "tasks" in result.output
    assert not (cli_env / "artifacts").exists()  # validation has no side effects


def test_cli_invalid_config_exit_code(cli_env: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\n")
    result = CliRunner().invoke(app, ["validate-config", "--config", str(bad)])
    assert result.exit_code == 2


def test_cli_dry_run_never_calls_provider(cli_env: Path) -> None:
    # local-qwen points at localhost:11434 which is not running; dry-run must still succeed.
    result = CliRunner().invoke(app, ["dry-run", "--model", "local-qwen", "--profile", "standard"])
    assert result.exit_code == 0, result.output
    assert "estimated wall time" in result.output
    remote = CliRunner().invoke(app, ["dry-run", "--model", "openai-paid"])
    assert remote.exit_code == 2 and "REMOTE" in remote.output


def test_cli_run_report_compare(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["run", "--model", "fake-reference", "--task", ",".join(FAST_TASKS)]
    )
    assert result.exit_code == 0, result.output
    run_id = result.output.split("run id: ")[1].split()[0]
    result2 = runner.invoke(
        app, ["run", "--model", "fake-noop", "--harness", "direct", "--task", FAST_TASKS[0]]
    )
    run_id2 = result2.output.split("run id: ")[1].split()[0]
    cmp = runner.invoke(app, ["compare", "--run-ids", f"{run_id},{run_id2}"])
    assert cmp.exit_code == 0 and "fake-noop | direct" in cmp.output
    for fmt in ("json", "csv", "html"):
        out = cli_env / f"r.{fmt}"
        res = runner.invoke(app, ["report", "--run-id", run_id, "--format", fmt, "--out", str(out)])
        assert res.exit_code == 0 and out.stat().st_size > 0
    assert runner.invoke(app, ["show", "--run-id", run_id, "--checks"]).exit_code == 0
    assert runner.invoke(app, ["show", "--run-id", "nope"]).exit_code == 2
    exported = cli_env / "export.json"
    assert (
        runner.invoke(app, ["export-run", "--run-id", run_id, "--out", str(exported)]).exit_code
        == 0
    )
    imp = runner.invoke(app, ["import-run", "--file", str(exported)])
    assert imp.exit_code == 0 and "2 evaluated" in imp.output


def test_cli_task_validate(cli_env: Path) -> None:
    result = CliRunner().invoke(
        app, ["task", "validate", str(REPO / "tasks/python/fix_interval_merge")]
    )
    assert result.exit_code == 0 and "behaviour OK" in result.output
