"""Performance probing: endpoint discovery, server info, calibration, counters, profiling."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from engineering_llm_bench import DATA_SCHEMA_VERSION
from engineering_llm_bench.byok.session import TaskSubmission, estimate_timing
from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.perf import (
    ProbeRefusedError,
    counter_delta,
    discover_endpoints,
    headline,
    load_jsonc,
    normalize_base_url,
    parse_prometheus,
    probe,
)
from engineering_llm_bench.perf.capture import assemble, resolve_endpoint
from engineering_llm_bench.reporting.stats import time_profile
from engineering_llm_bench.storage.db import MIGRATIONS, Store
from engineering_llm_bench.tools.runner import CommandRunner
from tests.fake_model_servers import MODEL, FakeOllama, FakeVllm, serve

SETTINGS = """{
  // VS Code user settings (JSONC)
  "editor.fontSize": 13,
  "github.copilot.chat.byok.ollamaEndpoint": "http://127.0.0.1:11500", /* custom port */
  "github.copilot.chat.customOAIModels": {
    "strata/qwen-3.8": {"name": "Qwen 3.8", "url": "http://localhost:8000/v1/chat/completions",},
  },
  "note": "http://example.com/not-a-model is still listed",
}"""


@pytest.mark.parametrize(
    ("url", "base"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://localhost:11434/api/chat", "http://localhost:11434"),
        ("http://127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1:8000"),
        ("https://gpu-box.local:8443/v1/", "https://gpu-box.local:8443"),
    ],
)
def test_normalize_base_url(url: str, base: str) -> None:
    assert normalize_base_url(url) == base


def test_discover_from_jsonc(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(SETTINGS)
    found = {c.base_url: c for c in discover_endpoints(load_jsonc(path))}
    assert "http://127.0.0.1:11500" in found
    assert "strata/qwen-3.8" in found["http://localhost:8000"].model_ids
    assert "http://localhost:11434" not in found  # Ollama URL given explicitly: no implicit default
    implicit = discover_endpoints({"providers": {"ollama": {"enabled": True}}})
    assert [c.base_url for c in implicit] == ["http://localhost:11434"]


def test_resolve_endpoint(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(SETTINGS)
    with pytest.raises(ConfigError, match="several endpoints"):
        resolve_endpoint(None, path, None)
    base, hints = resolve_endpoint(None, path, "strata/qwen-3.8")
    assert base == "http://localhost:8000" and hints[0] == "strata/qwen-3.8"
    assert resolve_endpoint("http://localhost:11434/v1", None, "m")[0] == "http://localhost:11434"
    with pytest.raises(ConfigError):
        resolve_endpoint(None, None, None)


def test_parse_prometheus_and_delta() -> None:
    text = (
        "# TYPE vllm:prompt_tokens_total counter\n"
        'vllm:prompt_tokens_total{model_name="a"} 100\n'
        'vllm:prompt_tokens_total{model_name="b"} 50\n'
        "vllm:generation_tokens_total 20\nunrelated_metric 5\n"
    )
    counters = parse_prometheus(text)
    assert counters == {"vllm:prompt_tokens_total": 150.0, "vllm:generation_tokens_total": 20.0}
    after = {"vllm:prompt_tokens_total": 400.0, "vllm:generation_tokens_total": 90.0}
    assert counter_delta(counters, after) == {
        "vllm:prompt_tokens_total": 250.0,
        "vllm:generation_tokens_total": 70.0,
    }
    assert counter_delta(after, counters) is None  # server restarted
    assert counter_delta(None, after) is None


def test_probe_ollama_uses_server_side_timings() -> None:
    FakeOllama.generate_calls = 0
    with serve(FakeOllama) as url:
        report = probe(url + "/api", calibrate_runs=2)
    h = headline(report)
    assert h["server"] == "ollama" and h["model"] == MODEL
    assert h["quantization"] == "Q4_K_M" and h["parameter_size"] == "8.2B"
    assert h["context_length"] == 40960 and h["gpu_fraction"] == 0.75
    assert h["gen_tps"] == pytest.approx(40.0) and h["cold_load_s"] == pytest.approx(1.5)
    cal = report["calibration"]
    assert cal["prompts"]["long"]["timing_source"] == "server" and cal["prompts"]["short"]["n"] == 2
    assert "template" not in str(report["server"].get("model"))  # large fields not stored


def test_probe_vllm_counters_and_client_timing() -> None:
    with serve(FakeVllm) as url:
        before = probe(url, calibrate_runs=0)
        session_traffic = probe(url + "/v1", calibrate_runs=1, max_tokens=64)
        after = probe(url, calibrate_runs=1, max_tokens=64)
    assert before["server"]["server"] == "vllm" and "calibration" not in before
    assert session_traffic["calibration"]["prompts"]["short"]["timing_source"] == "client"
    perf = assemble(url, before, after)
    delta = perf["session_counter_delta"]
    # Counters are read BEFORE the end probe's own calibration, so only the traffic in between
    # (here: one calibrating probe = warm-up + 2 prompts) is attributed to the session.
    assert delta["vllm:generation_tokens_total"] == 16 + 64 + 64
    assert headline(perf)["session_tokens"] == delta


def test_probe_refuses_remote_and_handles_unreachable() -> None:
    with pytest.raises(ProbeRefusedError):
        probe("https://api.example.com/v1", calibrate_runs=0)
    report = probe("http://127.0.0.1:9", calibrate_runs=1)
    assert report["server"]["reachable"] is False and "calibration" not in report


def test_command_rusage(tmp_path: Path) -> None:
    res = CommandRunner(LimitsConfig()).run(
        ["python", "-c", "x = bytearray(150 * 1024 * 1024); sum(range(3 * 10**6))"],
        cwd=tmp_path,
        root=tmp_path,
        scratch=tmp_path / "s",
        allowed={"python"},
    )
    if sys.platform != "linux":
        pytest.skip("rusage values checked on Linux")
    assert res.ok and res.cpu_s is not None and res.cpu_s > 0
    assert res.max_rss_mb is not None and res.max_rss_mb >= 150


def test_time_profile() -> None:
    rows = [
        {"model_time_s": 10.0, "agent_wall_s": 15.0, "output_tokens": 400, "eval_wall_s": 2.0},
        {"model_time_s": 20.0, "agent_wall_s": 26.0, "output_tokens": 400, "eval_wall_s": 4.0},
        {"model_time_s": None, "agent_wall_s": 5.0, "output_tokens": None, "eval_wall_s": None},
    ]
    t = time_profile(rows)
    assert t["model_s_median"] == 15.0 and t["tools_and_overhead_s_median"] == 5.5
    assert t["e2e_output_tps_median"] == 30.0 and t["e2e_output_tps_n"] == 2


def test_estimate_timing_from_mtimes() -> None:
    subs = [
        TaskSubmission(
            "a", "f1", True, first_change_ts=105.0, last_change_ts=110.0, result_ts=120.0
        ),
        TaskSubmission("b", "f2", True, first_change_ts=130.0, last_change_ts=150.0),
        TaskSubmission("c", "f3", True),
    ]
    t = estimate_timing(subs, started_at=100.0)
    assert [x["estimated_s"] for x in t["tasks"]] == [20.0, 30.0, None]
    assert t["observed_span_s"] == 50.0
    assert estimate_timing(subs, None)["tasks"][0]["estimated_s"] == 15.0


def test_migration_v1_to_v2(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT)")
    for stmt in MIGRATIONS[1].split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.execute("INSERT INTO schema_migrations VALUES (1, 'x')")
    conn.commit()
    conn.close()
    store = Store(db)
    assert store.schema_version() == DATA_SCHEMA_VERSION == 2
    cols = {r["name"] for r in store.query("PRAGMA table_info(attempts)")}
    assert {"model_time_s", "eval_cpu_s", "eval_max_rss_mb"} <= cols
