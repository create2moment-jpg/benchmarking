"""Server statistics over SSH / locally: config, scripts, parsing, monitor summary, profile."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from engineering_llm_bench.perf import remote
from engineering_llm_bench.perf.remote import (
    LocalShell,
    ServerAccess,
    SshConfig,
    SshShell,
    build_profile,
    go_duration_s,
    log_script,
    parse_sample,
    parse_server_log,
    summarize_requests,
    summarize_samples,
)
from tests.fake_model_servers import MODEL, FakeOllama, serve

OLLAMA_LOG = """\
time=2026-10-15T21:00:00.000+02:00 level=INFO source=server.go msg="llama runner started"
[GIN] 2026/10/15 - 21:00:10 | 200 |  4.5s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2026/10/15 - 21:00:30 | 200 |   1m2.5s |       127.0.0.1 | POST     "/v1/chat/completions"
[GIN] 2026/10/15 - 21:00:31 | 200 |     512.3µs |       127.0.0.1 | GET      "/api/ps"
[GIN] 2026/10/15 - 21:01:00 | 500 |   250ms |       127.0.0.1 | POST     "/api/chat"
"""
VLLM_LOG = (
    "INFO 10-15 21:00:10 [loggers.py:123] Engine 000: Avg prompt throughput: 812.4 tokens/s, "
    "Avg generation throughput: 41.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs\n"
    "INFO 10-15 21:00:20 Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 "
    "tokens/s, Running: 0 reqs\n"
)
FAKE_SMI = """#!/bin/sh
case "$*" in
  *index*) echo "0, NVIDIA RTX 4090, 87, 20000, 24564, 310.5"
           echo "1, NVIDIA RTX 4090, 3, 500, 24564, 40.0" ;;
  *) echo "NVIDIA RTX 4090, 550.54, 24564 MiB" ;;
esac
"""


def fake_bin(tmp_path: Path, name: str, body: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    path = bindir / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bindir


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("4.5s", 4.5), ("1m2.5s", 62.5), ("250ms", 0.25), ("512.3µs", 0.0005123), ("1h0m1s", 3601.0)],
)
def test_go_duration(text: str, seconds: float) -> None:
    assert go_duration_s(text) == pytest.approx(seconds)
    assert go_duration_s("abc") is None


def test_config_validation() -> None:
    with pytest.raises(ValueError, match=r"ssh\.target"):
        SshConfig(target="-oProxyCommand=evil")
    with pytest.raises(ValueError, match="Name=value"):
        SshConfig(target="u@h", options=["ProxyCommand evil"])
    with pytest.raises(ValueError, match="http"):
        ServerAccess.model_validate({"server": {"url": "file:///etc/passwd"}})
    with pytest.raises(ValueError):
        ServerAccess.model_validate({"bogus": 1})


def test_parse_ollama_and_vllm_logs() -> None:
    parsed = parse_server_log(OLLAMA_LOG + VLLM_LOG, "+0200")
    reqs = parsed["requests"]
    assert [r["path"] for r in reqs] == ["/api/chat", "/v1/chat/completions", "/api/chat"]
    assert reqs[0]["t_end"] == pytest.approx(1792090810.0)  # 21:00:10 +02:00
    summary = summarize_requests(parsed)
    assert summary is not None
    assert summary["count"] == 3 and summary["errors"] == 1
    assert summary["model_time_s"] == pytest.approx(67.2, abs=0.1)
    assert summary["first_start_t"] == pytest.approx(1792090810.0 - 4.5)
    assert summary["vllm_gen_tps_avg"] == 41.0  # idle lines (0 tok/s) are ignored


def test_log_script_custom_and_auto() -> None:
    custom = ServerAccess.model_validate(
        {"server": {"kind": "vllm", "log_command": "docker logs --since {since} vllm"}}
    )
    assert "docker logs --since 1700000000 vllm" in (log_script(custom, 1700000000.4) or "")
    assert log_script(ServerAccess.model_validate({"server": {"kind": "vllm"}}), 1.0) is None
    auto = log_script(ServerAccess(), 5.0) or ""
    assert 'journalctl -u ollama --since "@5"' in auto and "{{.Names}}" in auto


def test_ssh_argv_and_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stand-in `ssh`: records its arguments and runs the script from stdin locally.
    bindir = fake_bin(tmp_path, "ssh", f'#!/bin/sh\necho "$@" > {tmp_path}/argv\nexec bash -s\n')
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    shell = SshShell(SshConfig(target="me@gpu", port=2222, options=["ProxyJump=bastion"]))
    assert shell.run("echo hello") == "hello\n"
    argv = (tmp_path / "argv").read_text()
    assert "BatchMode=yes" in argv and "-p 2222" in argv and "ProxyJump=bastion" in argv
    assert argv.strip().endswith("-- me@gpu bash -s")


def test_snapshot_and_sample_against_fake_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindir = fake_bin(tmp_path, "nvidia-smi", FAKE_SMI)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    with serve(FakeOllama) as url:
        access = ServerAccess.model_validate(
            {"server": {"url": url}, "monitor": {"extra_commands": ["echo custom-status"]}}
        )
        snap = remote.take_snapshot(access, LocalShell())
        sample = remote.take_sample(access, LocalShell())
    assert snap["gpus"] == ["NVIDIA RTX 4090, 550.54, 24564 MiB"]
    assert snap["api_version"] == {"version": "0.12.3"}
    assert snap["models"][0]["name"] == MODEL and snap["models"][0]["quantization_level"]
    assert "custom-status" in snap["extra"]["CMD0"]
    assert [g["util_pct"] for g in sample["gpus"]] == [87.0, 3.0]
    assert sample["loaded"][0]["name"] == MODEL and sample["loaded"][0]["size_vram"]
    assert sample["mem_used_mb"] is None or sample["mem_used_mb"] > 0


def _sample(t: float, util: float, running: float = 0, gen: float = 0) -> dict[str, object]:
    text = (
        f"@@T\n{t}\n@@GPU\n0, GPU, {util}, 1000, 24000, 200\n@@PS\n{{}}\n@@METRICS\n"
        f'vllm:num_requests_running{{m="x"}} {running}\n'
        f'vllm:generation_tokens_total{{m="x"}} {gen}\n'
        f'vllm:prompt_tokens_total{{m="x"}} {gen * 10}\n@@END\n'
    )
    return parse_sample(text)


def test_summary_and_profile() -> None:
    samples = [
        _sample(100, 0),
        _sample(105, 90, 1, 100),
        _sample(110, 95, 1, 400),
        _sample(115, 5, 0, 600),
        {"t": 120, "error": "ConnectionError: x"},
    ]
    s = summarize_samples(samples)
    assert s["samples"] == 5 and s["sample_errors"] == 1
    assert s["gpu"]["util_peak_pct"] == 95 and s["gpu"]["busy_s"] == 10
    assert s["requests_running"]["first_t"] == 105 and s["requests_running"]["active_s"] == 10
    assert s["counter_delta"]["vllm:generation_tokens_total"] == 600
    p = build_profile(
        setup_at=50,
        finished_at=200,
        timing={"tasks": [{"task_id": "a", "estimated_s": 12.0}]},
        samples=s,
        requests=None,
        start={"hostname": "gpu1", "where": "vllm at x"},
        end=None,
        submissions_first_change=107,
        submissions_last_change=130,
    )
    assert p["agent_start_t"] == 105 and p["duration_source"] == "requests running"
    assert p["agent_duration_s"] == 25  # 105 -> last file change at 130
    assert p["setup_to_finish_s"] == 150
    assert p["generated_tokens"] == 600 and p["prompt_tokens"] == 6000
    assert p["gen_tps_server"] == 60.0  # 600 tokens / 10 s of requests running
    assert p["server_host"] == "gpu1" and p["tasks"][0]["task_id"] == "a"
    json.dumps(p)  # stored as JSON


def test_profile_prefers_server_log() -> None:
    req = summarize_requests(parse_server_log(OLLAMA_LOG, "+0000"))
    p = build_profile(
        setup_at=None, finished_at=0, timing=None, samples={}, requests=req, start=None, end=None
    )
    assert p["duration_source"] == "server log" and p["llm_requests"] == 3
    # first request started 21:00:30 - 62.5 s = 20:59:27.5; last ended 21:01:00
    assert p["agent_duration_s"] == pytest.approx(92.5)
