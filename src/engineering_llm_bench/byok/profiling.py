"""Time and server profiling of a BYOK session: a background monitor started by `byok start`,
stopped (plus end snapshot and server log) by `byok finish`, summarised into the run's
performance record when the session is scored."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from engineering_llm_bench.byok.session import (
    TaskSubmission,
    read_trusted_manifest,
    update_trusted_manifest,
)
from engineering_llm_bench.perf.remote import (
    MonitorConfig,
    ServerAccess,
    ServerConfig,
    build_profile,
    read_samples,
    read_server_log,
    summarize_requests,
    summarize_samples,
    take_snapshot,
)


def monitor_file(artifacts_dir: Path, session_id: str) -> Path:
    return artifacts_dir / "byok_monitor" / f"{session_id}.jsonl"


def access_for(config: ServerAccess | None, endpoint: str | None) -> ServerAccess | None:
    """server.yaml if given; otherwise the endpoint Copilot uses, polled over HTTP from here
    (no GPU sampling: a local URL may be an SSH tunnel to another machine)."""
    if config is not None:
        return config
    if endpoint:
        return ServerAccess(
            server=ServerConfig(url=endpoint.removesuffix("/v1")),
            monitor=MonitorConfig(gpu=False),
        )
    return None


def start_profiling(
    artifacts_dir: Path, session_id: str, access: ServerAccess | None
) -> dict[str, Any]:
    """Start snapshot + background monitor. Returns what was started (for the user)."""
    remote: dict[str, Any] = {"setup_at": time.time()}
    info: dict[str, Any] = {}
    if access is not None:
        remote["access"] = access.model_dump()
        remote["start"] = take_snapshot(access)
        info["where"] = access.describe()
        info["error"] = remote["start"].get("error")
        log = monitor_file(artifacts_dir, session_id).with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as fh:
            proc = subprocess.Popen(  # noqa: S603 - our own CLI, fixed arguments
                [
                    sys.executable,
                    "-c",
                    "from engineering_llm_bench.cli.main import main; main()",
                    "byok",
                    "monitor",
                    session_id,
                ],
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=fh,
                start_new_session=True,
            )
        remote["monitor_pid"] = proc.pid
        info["pid"] = proc.pid
    update_trusted_manifest(artifacts_dir, session_id, "remote", remote)
    return info


def _is_our_monitor(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return sys.platform == "darwin"  # no /proc: trust the recorded pid
    return b"byok monitor" in cmdline


def stop_profiling(artifacts_dir: Path, session_id: str) -> dict[str, Any] | None:
    """Stop the monitor and record the end snapshot and the server log (idempotent)."""
    manifest = read_trusted_manifest(artifacts_dir, session_id) or {}
    remote: dict[str, Any] = manifest.get("remote") or {}
    if not remote or remote.get("stopped_at"):
        return remote or None
    pid = remote.get("monitor_pid")
    if pid and _is_our_monitor(int(pid)):
        with contextlib.suppress(OSError):
            os.kill(int(pid), signal.SIGTERM)
    remote["stopped_at"] = time.time()
    if remote.get("access"):
        access = ServerAccess.model_validate(remote["access"])
        remote["end"] = take_snapshot(access)
        offset = (remote.get("start") or {}).get("utc_offset")
        remote["log"] = summarize_requests(
            read_server_log(access, remote.get("setup_at") or remote["stopped_at"], offset)
        )
    update_trusted_manifest(artifacts_dir, session_id, "remote", remote)
    return remote


def session_profile(
    artifacts_dir: Path,
    session_id: str,
    timing: dict[str, Any] | None,
    submissions: list[TaskSubmission],
) -> dict[str, Any]:
    manifest = read_trusted_manifest(artifacts_dir, session_id) or {}
    remote = manifest.get("remote") or {}
    perf = manifest.get("perf") or {}
    firsts = [s.first_change_ts for s in submissions if s.first_change_ts]
    lasts = [t for s in submissions for t in (s.result_ts, s.last_change_ts) if t]
    return build_profile(
        setup_at=remote.get("setup_at") or perf.get("started_at_epoch"),
        finished_at=remote.get("stopped_at") or time.time(),
        timing=timing,
        samples=summarize_samples(read_samples(monitor_file(artifacts_dir, session_id))),
        requests=remote.get("log"),
        start=remote.get("start"),
        end=remote.get("end"),
        submissions_first_change=min(firsts) if firsts else None,
        submissions_last_change=max(lasts) if lasts else None,
    )


def describe_profile(p: dict[str, Any]) -> list[str]:
    """Human-readable lines for the terminal."""

    def mins(s: float | None) -> str:
        return "unknown" if s is None else f"{s / 60:.1f} min"

    lines = [
        f"agent working time   {mins(p.get('agent_duration_s'))}"
        + (f"  (from {p['duration_source']})" if p.get("duration_source") else ""),
        f"setup -> finish      {mins(p.get('setup_to_finish_s'))}",
    ]
    if p.get("llm_requests") is not None:
        lines.append(
            f"LLM requests         {p['llm_requests']}"
            f"  (p50 {p.get('latency_p50_s')} s, p90 {p.get('latency_p90_s')} s)"
        )
    if p.get("model_time_s") is not None:
        share = p.get("model_time_share")
        lines.append(
            f"model busy           {mins(p['model_time_s'])}"
            + (f"  ({share * 100:.0f}% of working time)" if share is not None else "")
        )
    if p.get("generated_tokens"):
        lines.append(
            f"tokens               {p.get('prompt_tokens') or '?'} in / {p['generated_tokens']} out"
        )
    if p.get("gpu_util_peak_pct") is not None:
        lines.append(
            f"GPU                  avg {p.get('gpu_util_avg_pct')}%"
            f" (busy {p.get('gpu_util_busy_avg_pct')}%)"
            f", peak {p['gpu_util_peak_pct']}%, VRAM peak {p.get('gpu_mem_peak_mb')} of "
            f"{p.get('gpu_mem_total_mb')} MB"
        )
    if p.get("server_where"):
        lines.append(f"server               {p['server_where']}")
    return lines
