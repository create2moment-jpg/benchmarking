"""Model-server statistics collected on the machine that serves the model - over SSH when the
model runs elsewhere (``server.yaml``), or locally.

Everything is read-only: fixed shell snippets that read GPU state (``nvidia-smi``), load, the
server's own HTTP endpoints (queried *from* that machine with curl) and its log. No prompts are
sent and nothing is changed on the server.

* ``snapshot``: once at session start and end (host, GPUs, server version, models, extra
  commands from the config).
* ``sample``: every few seconds during the session, by a background monitor process (GPU
  utilisation / memory / power, loaded models, request gauges and token counters).
* ``server log``: at the end, the server's log since the session started; Ollama's request
  lines give the number of LLM requests and each request's duration, vLLM's periodic stats give
  throughput.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engineering_llm_bench.config.loader import ConfigError, read_yaml

SERVER_CONFIG_FILE = "server.yaml"
MAX_OUTPUT_BYTES = 4_000_000


# ---------------------------------------------------------------------------------- config
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SshConfig(_Strict):
    target: str = Field(description="user@host or an alias from ~/.ssh/config")
    port: int | None = Field(default=None, ge=1, le=65535)
    identity_file: str | None = None
    options: list[str] = Field(default_factory=list, description="extra `-o` options")

    @field_validator("target")
    @classmethod
    def _target(cls, v: str) -> str:
        if not v or v.startswith("-") or any(c.isspace() for c in v):
            raise ValueError("ssh.target must look like user@host or a host alias")
        return v

    @field_validator("options")
    @classmethod
    def _options(cls, v: list[str]) -> list[str]:
        for opt in v:
            if not re.fullmatch(r"[A-Za-z]+=[^\s]+", opt):
                raise ValueError(f"ssh option {opt!r} must look like Name=value")
        return v


class ServerConfig(_Strict):
    kind: Literal["auto", "ollama", "vllm", "llamacpp", "strata", "openai"] = "auto"
    url: str = Field(
        default="http://127.0.0.1:11434",
        description="the model server's URL as seen ON the server machine",
    )
    log_command: str | None = Field(
        default="auto",
        description="prints the server log since {since} (epoch s); auto = Ollama journal/docker/"
        "file; null = do not read logs",
    )

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not re.fullmatch(r"https?://[^\s'\"]+", v):
            raise ValueError("server.url must be an http(s) URL")
        return v.rstrip("/")


class MonitorConfig(_Strict):
    interval_s: float = Field(default=5.0, ge=1, le=300)
    gpu: bool = True
    extra_commands: list[str] = Field(
        default_factory=list, description="run at start and end; output is recorded"
    )


class ServerAccess(_Strict):
    ssh: SshConfig | None = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    def describe(self) -> str:
        where = f"ssh {self.ssh.target}" if self.ssh else "this machine"
        return f"{self.server.kind} at {self.server.url} on {where}"


def load_server_access(path: Path) -> ServerAccess | None:
    if not path.is_file():
        return None
    data = read_yaml(path) or {}
    try:
        return ServerAccess.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


# ---------------------------------------------------------------------------------- shells
class Shell(Protocol):
    def run(self, script: str, timeout: float = 30.0) -> str: ...


def _run(argv: list[str], script: str, timeout: float) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv; the script goes to stdin
        argv,
        input=script.encode(),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode == 255 and argv[0] == "ssh":
        raise ConnectionError(proc.stderr.decode(errors="replace").strip()[-500:] or "ssh failed")
    return proc.stdout[:MAX_OUTPUT_BYTES].decode(errors="replace")


class LocalShell:
    """Runs the snippets on this machine (model served locally)."""

    def run(self, script: str, timeout: float = 30.0) -> str:
        return _run(["bash", "-s"], script, timeout)


class SshShell:
    """Runs the snippets on the server over SSH (key-based, never prompts). A shared control
    connection keeps per-sample overhead low."""

    def __init__(self, cfg: SshConfig) -> None:
        self.cfg = cfg
        ctl = Path(tempfile.gettempdir()) / f"elb-ssh-{os.getuid()}"
        ctl.mkdir(mode=0o700, exist_ok=True)
        self.argv = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={ctl}/%C",
            "-o", "ControlPersist=120",
        ]  # fmt: skip
        if cfg.port:
            self.argv += ["-p", str(cfg.port)]
        if cfg.identity_file:
            self.argv += ["-i", str(Path(cfg.identity_file).expanduser())]
        for opt in cfg.options:
            self.argv += ["-o", opt]
        self.argv += ["--", cfg.target, "bash -s"]

    def run(self, script: str, timeout: float = 30.0) -> str:
        return _run(self.argv, script, timeout)


def shell_for(access: ServerAccess) -> Shell:
    return SshShell(access.ssh) if access.ssh else LocalShell()


# ---------------------------------------------------------------------------------- scripts
_PRELUDE = """\
get() { curl -s -m 3 "$1" 2>/dev/null || wget -qO- -T 3 "$1" 2>/dev/null; }
"""


def sample_script(access: ServerAccess) -> str:
    url = shlex.quote(access.server.url)
    gpu = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null"
        if access.monitor.gpu
        else ":"
    )
    return (
        _PRELUDE
        + f"""\
echo "@@T"; date +%s.%N
echo "@@GPU"; {gpu}
echo "@@LOAD"; cat /proc/loadavg 2>/dev/null
echo "@@MEM"; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo 2>/dev/null
echo "@@PS"; get {url}/api/ps; echo
echo "@@METRICS"; get {url}/metrics | grep -E '^(vllm|llamacpp)[:_]' | head -n 400
echo "@@END"
"""
    )


def snapshot_script(access: ServerAccess) -> str:
    url = shlex.quote(access.server.url)
    extra = "".join(
        f'echo "@@CMD{i}"; ( {cmd} ) 2>&1 | head -c 4000; echo\n'
        for i, cmd in enumerate(access.monitor.extra_commands)
    )
    return (
        _PRELUDE
        + f"""\
echo "@@T"; date +%s.%N
echo "@@HOST"; hostname; uname -sr; date +%z
echo "@@SMI"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  2>/dev/null
echo "@@OLLAMA"; ollama --version 2>/dev/null | head -n 2
echo "@@APIVERSION"; get {url}/api/version; echo
echo "@@VERSION"; get {url}/version; echo
echo "@@TAGS"; get {url}/api/tags | head -c 20000; echo
echo "@@MODELS"; get {url}/v1/models | head -c 20000; echo
echo "@@PROPS"; get {url}/props | head -c 4000; echo
{extra}echo "@@END"
"""
    )


_OLLAMA_LOG_AUTO = (
    'out=$(journalctl -u ollama --since "@{since}" --no-pager -o cat 2>/dev/null | tail -n 50000); '
    'if [ -n "$out" ]; then echo "$out"; '
    "elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx ollama; then "
    "docker logs --since {since} ollama 2>&1 | tail -n 50000; "
    'elif [ -f "$HOME/.ollama/logs/server.log" ]; then '
    'tail -n 50000 "$HOME/.ollama/logs/server.log"; fi'
)


def log_script(access: ServerAccess, since_epoch: float) -> str | None:
    cmd = access.server.log_command
    if cmd is None:
        return None
    if cmd == "auto":
        if access.server.kind not in ("auto", "ollama"):
            return None
        cmd = _OLLAMA_LOG_AUTO
    return f"( {cmd.replace('{since}', str(int(since_epoch)))} ) 2>&1 | tail -n 50000\n"


# ---------------------------------------------------------------------------------- parsing
def sections(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("@@") and re.fullmatch(r"@@[A-Z0-9]+", line.strip()):
            current = line.strip()[2:]
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def _float(v: str) -> float | None:
    try:
        return float(v)
    except ValueError:
        return None


def _json(text: str) -> Any:
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


_GAUGE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)")


def parse_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        m = _GAUGE.match(line)
        if m and not m.group(1).endswith("_bucket"):
            value = _float(m.group(3))
            if value is not None:
                totals[m.group(1)] = totals.get(m.group(1), 0.0) + value
    return totals


def parse_sample(text: str) -> dict[str, Any]:
    s = sections(text)
    gpus = []
    for line in s.get("GPU", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "util_pct": _float(parts[2]),
                    "mem_used_mb": _float(parts[3]),
                    "mem_total_mb": _float(parts[4]),
                    "power_w": _float(parts[5]) if len(parts) > 5 else None,
                }
            )
    mem = dict(re.findall(r"^(\w+):\s+(\d+)", s.get("MEM", ""), re.M))
    ps = _json(s.get("PS", ""))
    loaded = [
        {"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram")}
        for m in ((ps or {}).get("models") or [])
        if isinstance(m, dict)
    ]
    load = s.get("LOAD", "").split()
    return {
        "t": _float(s.get("T", "")) or time.time(),
        "gpus": gpus,
        "load1": _float(load[0]) if load else None,
        "mem_used_mb": (int(mem["MemTotal"]) - int(mem["MemAvailable"])) / 1024
        if "MemTotal" in mem and "MemAvailable" in mem
        else None,
        "loaded": loaded,
        "metrics": parse_metrics(s.get("METRICS", "")),
    }


def parse_snapshot(text: str) -> dict[str, Any]:
    s = sections(text)
    host = s.get("HOST", "").splitlines()
    return {
        "t": _float(s.get("T", "")),
        "hostname": host[0] if host else None,
        "os": host[1] if len(host) > 1 else None,
        "utc_offset": host[2] if len(host) > 2 else None,
        "gpus": [line.strip() for line in s.get("SMI", "").splitlines() if line.strip()],
        "ollama_version": s.get("OLLAMA") or None,
        "api_version": _json(s.get("APIVERSION", "")),
        "version": _json(s.get("VERSION", "")),
        "models": [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                **{k: v for k, v in (m.get("details") or {}).items() if k != "families"},
            }
            for m in ((_json(s.get("TAGS", "")) or {}).get("models") or [])
            if isinstance(m, dict)
        ]
        or [m.get("id") for m in ((_json(s.get("MODELS", "")) or {}).get("data") or [])],
        "props": _json(s.get("PROPS", "")),
        "extra": {k: v for k, v in s.items() if k.startswith("CMD")},
    }


_GO_DURATION = re.compile(r"(\d+(?:\.\d+)?)(h|ms|µs|us|ns|m|s)")
_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9}


def go_duration_s(text: str) -> float | None:
    parts = _GO_DURATION.findall(text.strip())
    if not parts or "".join(n + u for n, u in parts) != text.strip():
        return None
    return sum(float(n) * _UNITS[u] for n, u in parts)


_OLLAMA_REQ = re.compile(
    r"\[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2})\s+\|\s+(\d{3})\s+\|\s+(\S+)\s+\|"
    r"\s+\S+\s+\|\s+(\w+)\s+\"([^\"]+)\""
)
_GENERATION_PATHS = (
    "/api/chat",
    "/api/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
)
_VLLM_STATS = re.compile(
    r"Avg prompt throughput:\s*([\d.]+) tokens/s, Avg generation throughput:\s*([\d.]+) tokens/s"
    r"(?:, Running:\s*(\d+))?"
)


def _offset(utc_offset: str | None) -> timezone:
    m = re.fullmatch(r"([+-])(\d{2})(\d{2})", (utc_offset or "").strip())
    if not m:
        return UTC
    delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return timezone(delta if m.group(1) == "+" else -delta)


def parse_server_log(text: str, utc_offset: str | None = None) -> dict[str, Any]:
    """Generation requests from Ollama's request log; throughput lines from vLLM's log."""
    tz = _offset(utc_offset)
    requests = []
    for m in _OLLAMA_REQ.finditer(text):
        if m.group(4) != "POST" or m.group(5) not in _GENERATION_PATHS:
            continue
        duration = go_duration_s(m.group(3))
        when = datetime.strptime(m.group(1), "%Y/%m/%d - %H:%M:%S").replace(tzinfo=tz)
        requests.append(
            {
                "t_end": when.timestamp(),
                "status": int(m.group(2)),
                "duration_s": duration,
                "path": m.group(5),
            }
        )
    vllm = [
        {"prompt_tps": float(a), "gen_tps": float(b), "running": int(c) if c else None}
        for a, b, c in _VLLM_STATS.findall(text)
    ]
    return {"requests": requests, "vllm_stats": vllm, "lines": text.count("\n")}


# ---------------------------------------------------------------------------------- collecting
def take_snapshot(access: ServerAccess, shell: Shell | None = None) -> dict[str, Any]:
    shell = shell or shell_for(access)
    try:
        return {"where": access.describe(), **parse_snapshot(shell.run(snapshot_script(access)))}
    except (OSError, subprocess.SubprocessError, ConnectionError) as exc:
        return {"where": access.describe(), "error": f"{type(exc).__name__}: {exc}"}


def take_sample(access: ServerAccess, shell: Shell) -> dict[str, Any]:
    try:
        return parse_sample(shell.run(sample_script(access), timeout=20.0))
    except (OSError, subprocess.SubprocessError, ConnectionError) as exc:
        return {"t": time.time(), "error": f"{type(exc).__name__}: {exc}"}


def read_server_log(
    access: ServerAccess, since_epoch: float, utc_offset: str | None, shell: Shell | None = None
) -> dict[str, Any] | None:
    script = log_script(access, since_epoch)
    if script is None:
        return None
    try:
        text = (shell or shell_for(access)).run(script, timeout=60)
    except (OSError, subprocess.SubprocessError, ConnectionError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    parsed = parse_server_log(text, utc_offset)
    parsed["requests"] = [r for r in parsed["requests"] if r["t_end"] >= since_epoch - 1]
    return parsed


def run_monitor(
    access: ServerAccess, out: Path, *, max_hours: float = 12.0, shell: Shell | None = None
) -> None:
    """Append one sample per interval to ``out`` (JSON lines) until SIGTERM/SIGINT."""
    stop = False

    def _stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    shell = shell or shell_for(access)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max_hours * 3600
    interval = access.monitor.interval_s
    while not stop and time.time() < deadline:
        began = time.monotonic()
        sample = take_sample(access, shell)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample) + "\n")
        wait = interval * (4 if "error" in sample else 1) - (time.monotonic() - began)
        end = time.monotonic() + max(wait, 0.5)
        while not stop and time.monotonic() < end:
            time.sleep(0.25)


def read_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


# ---------------------------------------------------------------------------------- summary
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def _r(v: float | None, d: int = 1) -> float | None:
    return None if v is None else round(v, d)


def summarize_samples(samples: list[dict[str, Any]], busy_util_pct: float = 20.0) -> dict[str, Any]:
    ok = [s for s in samples if "error" not in s]
    out: dict[str, Any] = {"samples": len(samples), "sample_errors": len(samples) - len(ok)}
    if not ok:
        return out
    ts = [s["t"] for s in ok]
    out["first_t"], out["last_t"] = min(ts), max(ts)
    interval = (
        statistics.median([b - a for a, b in itertools.pairwise(ts)]) if len(ts) > 1 else None
    )
    gpu_rows = [g for s in ok for g in s.get("gpus") or []]
    if gpu_rows:
        utils = [g["util_pct"] for g in gpu_rows if g.get("util_pct") is not None]
        # Per-sample max over GPUs for "busy" detection and the activity timeline.
        per_sample = [max((g.get("util_pct") or 0.0) for g in s.get("gpus") or [{}]) for s in ok]
        busy = [s["t"] for s, u in zip(ok, per_sample, strict=False) if u >= busy_util_pct]
        out["gpu"] = {
            "names": sorted({g.get("name") or "?" for g in gpu_rows}),
            "util_avg_pct": _r(statistics.fmean(utils)) if utils else None,
            "util_busy_avg_pct": _r(statistics.fmean([u for u in per_sample if u >= busy_util_pct]))
            if busy
            else None,
            "util_peak_pct": _r(max(utils)) if utils else None,
            "mem_peak_mb": _r(max((g.get("mem_used_mb") or 0) for g in gpu_rows), 0),
            "mem_total_mb": _r(max((g.get("mem_total_mb") or 0) for g in gpu_rows), 0),
            "power_avg_w": _r(statistics.fmean(p))
            if (p := [g["power_w"] for g in gpu_rows if g.get("power_w") is not None])
            else None,
            "busy_s": _r(len(busy) * interval) if interval else None,
            "first_busy_t": min(busy) if busy else None,
            "last_busy_t": max(busy) if busy else None,
        }
    running = [
        (s["t"], s["metrics"].get("vllm:num_requests_running", 0.0)
         + s["metrics"].get("llamacpp:requests_processing", 0.0))
        for s in ok if s.get("metrics")
    ]  # fmt: skip
    if running:
        active = [t for t, n in running if n > 0]
        out["requests_running"] = {
            "active_s": _r(len(active) * interval) if interval else None,
            "first_t": min(active) if active else None,
            "last_t": max(active) if active else None,
            "peak": max(n for _, n in running),
        }
    first_m, last_m = ok[0].get("metrics") or {}, ok[-1].get("metrics") or {}
    delta = {
        k: round(last_m[k] - first_m.get(k, 0.0), 3)
        for k in last_m
        if re.search(r"(tokens_total|_sum|_count|tokens_predicted_total)$", k)
        and last_m[k] - first_m.get(k, 0.0) > 0
    }
    if delta:
        out["counter_delta"] = delta
    models = sorted({m["name"] for s in ok for m in s.get("loaded") or [] if m.get("name")})
    if models:
        out["loaded_models"] = models
        vram = [
            m["size_vram"] / m["size"]
            for s in ok
            for m in s.get("loaded") or []
            if m.get("size") and m.get("size_vram") is not None
        ]
        out["gpu_share_min"] = _r(min(vram), 3) if vram else None
    return out


def summarize_requests(log: dict[str, Any] | None) -> dict[str, Any] | None:
    if not log or "error" in log:
        return log
    reqs = [r for r in log.get("requests") or [] if r.get("duration_s") is not None]
    out: dict[str, Any] = {"log_lines": log.get("lines")}
    if reqs:
        durations = [r["duration_s"] for r in reqs]
        out.update(
            {
                "count": len(reqs),
                "errors": sum(1 for r in reqs if r["status"] >= 400),
                "model_time_s": _r(sum(durations)),
                "latency_p50_s": _r(_pct(durations, 0.5), 2),
                "latency_p90_s": _r(_pct(durations, 0.9), 2),
                "latency_max_s": _r(max(durations), 2),
                "first_start_t": min(r["t_end"] - r["duration_s"] for r in reqs),
                "last_end_t": max(r["t_end"] for r in reqs),
                "by_path": {
                    p: sum(1 for r in reqs if r["path"] == p) for p in {r["path"] for r in reqs}
                },
            }
        )
    stats = [s for s in log.get("vllm_stats") or [] if s["gen_tps"] > 0]
    if stats:
        out["vllm_gen_tps_avg"] = _r(statistics.fmean(s["gen_tps"] for s in stats))
        out["vllm_prompt_tps_avg"] = _r(statistics.fmean(s["prompt_tps"] for s in stats))
    return out


def build_profile(
    *,
    setup_at: float | None,
    finished_at: float,
    timing: dict[str, Any] | None,
    samples: dict[str, Any] | None,
    requests: dict[str, Any] | None,
    start: dict[str, Any] | None,
    end: dict[str, Any] | None,
    submissions_first_change: float | None = None,
    submissions_last_change: float | None = None,
) -> dict[str, Any]:
    """Run-level time profile: when the agent actually worked (from the server's request log,
    GPU activity or file timestamps, in that order of preference), how much of that the model
    was busy, and server resources."""
    samples = samples or {}
    req = requests if requests and "error" not in requests else {}
    gpu = samples.get("gpu") or {}
    running = samples.get("requests_running") or {}
    starts = [
        ("server log", req.get("first_start_t")),
        ("requests running", running.get("first_t")),
        ("GPU activity", gpu.get("first_busy_t")),
        ("file changes", submissions_first_change),
    ]
    ends = [
        ("server log", req.get("last_end_t")),
        ("requests running", running.get("last_t")),
        ("GPU activity", gpu.get("last_busy_t")),
        ("file changes", submissions_last_change),
    ]
    first: tuple[str | None, float | None] = next(
        ((src, t) for src, t in starts if t), (None, None)
    )
    if first[1] is not None and setup_at and first[1] < setup_at:
        first = (first[0], setup_at)  # the agent cannot have started before the session existed
    candidates_end = [t for _, t in ends if t]
    last_t = max(candidates_end) if candidates_end else None
    duration = (last_t - first[1]) if (first[1] and last_t and last_t >= first[1]) else None
    model_time = req.get("model_time_s") or running.get("active_s")
    tokens = samples.get("counter_delta") or {}
    gen = sum(
        v
        for k, v in tokens.items()
        if re.search(r"generation_tokens_total|tokens_predicted_total$", k)
    )
    prompt = sum(v for k, v in tokens.items() if re.search(r"prompt_tokens_total$", k))
    return {
        "setup_at": setup_at,
        "finished_at": finished_at,
        "setup_to_finish_s": _r(finished_at - setup_at) if setup_at else None,
        "agent_start_t": first[1],
        "agent_end_t": last_t,
        "agent_duration_s": _r(duration),
        "duration_source": first[0],
        "llm_requests": req.get("count"),
        "llm_request_errors": req.get("errors"),
        "model_time_s": model_time,
        "model_time_share": _r(model_time / duration, 3) if model_time and duration else None,
        "latency_p50_s": req.get("latency_p50_s"),
        "latency_p90_s": req.get("latency_p90_s"),
        "latency_max_s": req.get("latency_max_s"),
        "prompt_tokens": prompt or None,
        "generated_tokens": gen or None,
        "gen_tps_server": req.get("vllm_gen_tps_avg")
        or (_r(gen / model_time) if gen and model_time else None),
        "gpu_names": gpu.get("names"),
        "gpu_util_avg_pct": gpu.get("util_avg_pct"),
        "gpu_util_busy_avg_pct": gpu.get("util_busy_avg_pct"),
        "gpu_util_peak_pct": gpu.get("util_peak_pct"),
        "gpu_mem_peak_mb": gpu.get("mem_peak_mb"),
        "gpu_mem_total_mb": gpu.get("mem_total_mb"),
        "gpu_power_avg_w": gpu.get("power_avg_w"),
        "gpu_busy_s": gpu.get("busy_s"),
        "loaded_models": samples.get("loaded_models"),
        "gpu_share_min": samples.get("gpu_share_min"),
        "samples": samples.get("samples"),
        "sample_errors": samples.get("sample_errors"),
        "server_host": (end or start or {}).get("hostname"),
        "server_where": (end or start or {}).get("where"),
        "tasks": (timing or {}).get("tasks"),
        "requests_by_path": req.get("by_path"),
    }
