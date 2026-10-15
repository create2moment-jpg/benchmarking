"""Acceptance 8 and 9: a real OpenAI-compatible HTTP endpoint is configured with config only;
exports match stored results and are redacted; import/export round trips preserve known fields
and leave unknown fields unknown."""

from __future__ import annotations

import csv
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import ModelConfig
from engineering_llm_bench.domain import Profile
from engineering_llm_bench.harnesses.protocol import render_action
from engineering_llm_bench.importing import RunImporter, export_import_format
from engineering_llm_bench.orchestration.runner import RunRequest
from engineering_llm_bench.reporting import ReportOptions, build_report, to_csv, to_html, to_json
from engineering_llm_bench.workspace.manager import iter_files
from tests.conftest import FAST_TASKS, REPO, BenchEnv

pytestmark = pytest.mark.integration
TASK = "python.fix-interval-merge"


class ScriptedChatServer(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server that drives the agent protocol with the reference fix."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json({"data": [{"id": "scripted"}]})
        else:
            self._json({"error": "nope"}, 404)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ScriptedChatServer.requests.append(payload)
        step = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        ref = REPO / "tasks/python/fix_interval_merge/reference"
        plan = [{"action": "read_file", "path": "intervals.py"}]
        plan += [
            {
                "action": "write_file",
                "path": p.relative_to(ref).as_posix(),
                "content": p.read_text(),
            }
            for p in iter_files(ref)
        ]
        plan += [
            {"action": "run_command", "argv": ["pytest", "-q"]},
            {
                "action": "finish",
                "summary": "Fixed sorting; tested with pytest. Limitation: none.",
            },
        ]
        text = render_action(plan[min(step, len(plan) - 1)])
        self._json(
            {
                "id": f"chat-{step}",
                "model": "scripted-model-v1",
                "choices": [
                    {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100 + step, "completion_tokens": 20},
            }
        )


@pytest.fixture
def local_endpoint() -> Iterator[str]:
    ScriptedChatServer.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedChatServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_local_openai_compatible_endpoint(bench_env: BenchEnv, local_endpoint: str) -> None:
    model = ModelConfig.model_validate(
        {
            "alias": "local-scripted",
            "provider": "openai_compatible",
            "endpoint_type": "vllm",
            "base_url": local_endpoint,
            "model_id": "scripted",
            "serving_runtime": "test-server",
            "quantization": "none",
            "context_limit": 8192,
            "sampling": {"temperature": 0.0, "seed": 1},
            "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
        }
    )
    bench_env.models = bench_env.models.model_copy(
        update={"models": [*bench_env.models.models, model]}
    )
    run_id = bench_env.runner().run(
        RunRequest(
            model_alias="local-scripted", harness="agent", profile=Profile.SMOKE, task_ids=[TASK]
        )
    )
    a = bench_env.store.attempts_for_run(run_id)[0]
    assert a["passed"] == 1 and a["failure_class"] == "none"
    assert a["input_tokens"] > 0 and a["output_tokens"] > 0
    expected_cost = (a["input_tokens"] * 1.0 + a["output_tokens"] * 2.0) / 1e6
    assert a["cost_usd"] == pytest.approx(expected_cost)
    first = ScriptedChatServer.requests[0]
    assert first["model"] == "scripted" and first["temperature"] == 0.0 and first["seed"] == 1
    assert "metadata" not in first
    run_row = bench_env.store.get_run(run_id)
    assert run_row and json.loads(run_row["model_config_json"])["quantization"] == "none"


def make_runs(env: BenchEnv) -> list[str]:
    ids = []
    for alias in ("fake-reference", "fake-noop"):
        ids.append(
            env.runner().run(
                RunRequest(
                    model_alias=alias, harness="agent", profile=Profile.SMOKE, task_ids=FAST_TASKS
                )
            )
        )
    return ids


def test_exports_match_store_and_redact(
    bench_env: BenchEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ids = make_runs(bench_env)
    # Plant a secret in persisted data to prove exports redact it.
    planted = "sk-proj-THISSHOULDNEVERAPPEAR0123456789"
    aid = bench_env.store.attempts_for_run(run_ids[0])[0]["id"]
    bench_env.store.upsert_attempt({"id": aid, "summary": f"used key {planted}"})
    report = build_report(
        bench_env.store, bench_env.artifacts, bench_env.artifacts.redactor, ReportOptions()
    )
    text_json, text_csv, text_html = to_json(report), to_csv(report), to_html(report)
    for text in (text_json, text_csv, text_html):
        assert planted not in text
    rows = list(csv.DictReader(io.StringIO(text_csv)))
    stored = {a["id"]: a for a in bench_env.store.all_attempts()}
    assert len(rows) == len(stored) == 4
    by_key = {(r["run_id"], r["task_id"]): r for r in rows}
    for a in stored.values():
        row = by_key[(a["run_id"], a["task_id"])]
        assert float(row["score"]) == pytest.approx(a["score"])
        assert row["passed"] == str(a["passed"])
    parsed = json.loads(text_json)
    assert {g for g in parsed["groups"]} == {"fake-reference | agent", "fake-noop | agent"}
    assert parsed["groups"]["fake-reference | agent"]["pass_rate"] == 1.0
    assert parsed["groups"]["fake-noop | agent"]["pass_rate"] == 0.0
    assert "MOCK provider" in " ".join(parsed["groups"]["fake-noop | agent"]["warnings"])
    assert "trace" not in parsed["attempts"][0] and "diff" not in parsed["attempts"][0]
    opted = build_report(
        bench_env.store,
        bench_env.artifacts,
        bench_env.artifacts.redactor,
        ReportOptions(include_transcripts=True, include_source=True),
    )
    assert "trace" in opted["attempts"][0] and "diff" in opted["attempts"][0]
    assert "<svg" in text_html and "fake-reference" in text_html


def test_report_unknown_run_id(bench_env: BenchEnv) -> None:
    with pytest.raises(KeyError):
        build_report(
            bench_env.store,
            bench_env.artifacts,
            bench_env.artifacts.redactor,
            ReportOptions(run_ids=["nope"]),
        )


def write_import(path: Path, attempts: list[dict[str, Any]], **run: Any) -> Path:
    doc = {
        "schema_version": 1,
        "format": "engineering-llm-bench/import",
        "run": {"model_label": "copilot-byok-gpt", "harness": "copilot-byok", **run},
        "attempts": attempts,
    }
    path.write_text(json.dumps(doc))
    return path


def reference_patch(env: BenchEnv) -> str:
    run_id = env.runner().run(
        RunRequest(
            model_alias="fake-reference", harness="direct", profile=Profile.SMOKE, task_ids=[TASK]
        )
    )
    return export_import_format(env.store, env.artifacts, run_id)["attempts"][0]["patch"]


def test_import_evaluates_patch_and_keeps_unknowns_null(
    bench_env: BenchEnv, tmp_path: Path
) -> None:
    patch = reference_patch(bench_env)
    importer = RunImporter(
        bench_env.bench, bench_env.registry, bench_env.store, bench_env.artifacts
    )
    path = write_import(
        tmp_path / "imp.json",
        [
            {
                "task_id": TASK,
                "patch": patch,
                "elapsed_s": 42.0,
                "summary": "fixed",
                "self_reported_passed": True,
                "command_logs": [{"command": "rm -rf /", "exit_code": 0}],
            }
        ],
    )
    result = importer.import_file(path)
    a = bench_env.store.attempts_for_run(result.run_id)[0]
    assert a["passed"] == 1 and a["status"] == "completed"
    assert a["input_tokens"] is None and a["output_tokens"] is None and a["cost_usd"] is None
    assert a["turns"] is None and a["agent_wall_s"] == 42.0
    run_row = bench_env.store.get_run(result.run_id)
    assert run_row and run_row["mode"] == "manual" and run_row["source"] == "imported"
    assert run_row["remote"] is None and run_row["provider"] is None
    dims = {d["dimension"]: d["score"] for d in bench_env.store.dimension_scores(a["id"])}
    assert dims["workflow"] is None  # not observable for manual runs
    # Round trip: export -> same known fields, unknown stay null.
    exported = export_import_format(bench_env.store, bench_env.artifacts, result.run_id)
    item = exported["attempts"][0]
    assert item["patch"] == patch and item["elapsed_s"] == 42.0 and item["summary"] == "fixed"
    assert item["input_tokens"] is None and item["cost_usd"] is None
    assert (
        exported["run"]["model_label"] == "copilot-byok-gpt"
        and exported["run"]["provider_label"] is None
    )
    again = importer.import_file(
        write_import(
            tmp_path / "again.json",
            exported["attempts"],
            model_label="copilot-byok-gpt",
            harness="copilot-byok",
        )
    )
    assert bench_env.store.attempts_for_run(again.run_id)[0]["score"] == a["score"]


def test_import_rejects_unsafe_and_unknown(bench_env: BenchEnv, tmp_path: Path) -> None:
    importer = RunImporter(
        bench_env.bench, bench_env.registry, bench_env.store, bench_env.artifacts
    )
    with pytest.raises(ConfigError, match="unknown task"):
        importer.import_file(write_import(tmp_path / "a.json", [{"task_id": "python.nope"}]))
    with pytest.raises(ConfigError, match="Extra inputs"):
        importer.import_file(
            write_import(tmp_path / "b.json", [{"task_id": TASK, "tokens_guess": 5}])
        )
    evil = "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
    result = importer.import_file(
        write_import(tmp_path / "c.json", [{"task_id": TASK, "patch": evil}])
    )
    a = bench_env.store.attempts_for_run(result.run_id)[0]
    assert a["failure_class"] == "malformed_output" and "unsafe path" in a["error"]
    noeval = importer.import_file(
        write_import(tmp_path / "d.json", [{"task_id": TASK, "patch": evil}]),
        evaluate_patches=False,
    )
    assert bench_env.store.attempts_for_run(noeval.run_id)[0]["status"] == "skipped"
    with pytest.raises(ConfigError, match="patch_file"):
        importer.import_file(
            write_import(tmp_path / "e.json", [{"task_id": TASK, "patch_file": "../x.patch"}])
        )
