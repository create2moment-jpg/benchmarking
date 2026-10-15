"""Probe a model server directly: what is being served, how fast it is, and what it used.

Works for any server behind an IDE agent (Copilot BYOK) or the built-in harnesses:

* **Server info** - Ollama (``/api/version``, ``/api/tags``, ``/api/ps``, ``/api/show``:
  quantization, parameter count, context length, bytes resident in VRAM), OpenAI-compatible
  servers (``/v1/models``), vLLM (``/version``), llama.cpp (``/props``).
* **Counters** - Prometheus ``/metrics`` (vLLM, llama.cpp ``--metrics``): cumulative prompt /
  generation tokens, request counts, TTFT and latency sums. Snapshots before and after a session
  give the tokens that session consumed (``counter_delta``). Ollama has no such endpoint.
* **Calibration** - a fixed, repository-independent prompt set timed against the endpoint:
  time to first token, prompt-processing and generation tokens/s, cold-load time. Ollama's
  native API returns server-side nanosecond timings, which are preferred over client timing.
* **Compute** - while calibrating on a local server: GPU utilisation / memory peaks sampled from
  ``nvidia-smi`` when available, and load average.

Only GET requests and the fixed calibration prompts are sent; no repository content.
All numbers are measured, never estimated; anything unavailable is ``None``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from engineering_llm_bench.config.models import is_local_host
from engineering_llm_bench.perf.discovery import normalize_base_url
from engineering_llm_bench.storage.db import utcnow

PROBE_FORMAT = "engineering-llm-bench/perf-probe"
_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")
_INTERESTING = re.compile(
    r"(prompt_tokens|generation_tokens|tokens_predicted|request_success|requests_processing|"
    r"time_to_first_token_seconds_(sum|count)|e2e_request_latency_seconds_(sum|count)|"
    r"prompt_seconds|tokens_predicted_seconds|n_decode|n_busy_slots)"
)
_SHORT_PROMPT = (
    "Write a Python function `clamp(value, low, high)` that clamps a number to a closed range "
    "and raises ValueError when low > high. Include a docstring."
)
_LONG_CONTEXT = (
    "def merge_intervals(intervals):\n"
    "    merged = []\n"
    "    for start, end in sorted(intervals):\n"
    "        if merged and start <= merged[-1][1]:\n"
    "            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n"
    "        else:\n"
    "            merged.append((start, end))\n"
    "    return merged\n\n"
)
_LONG_PROMPT = (
    "Below is a module repeated several times. Summarise in three bullet points what the "
    "function does and one edge case it handles.\n\n" + _LONG_CONTEXT * 24
)
CALIBRATION_PROMPTS = {"short": _SHORT_PROMPT, "long": _LONG_PROMPT}


class ProbeRefusedError(Exception):
    """The endpoint is remote and probing it was not explicitly allowed."""


@dataclass
class Sample:
    prompt: str
    ttft_s: float | None
    total_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    prompt_tps: float | None
    gen_tps: float | None
    load_s: float | None
    timing_source: str  # "server" (Ollama native) | "client"
    error: str | None = None


@dataclass
class GpuSampler:
    """Background sampling of ``nvidia-smi`` (local GPUs only)."""

    interval_s: float = 0.25
    samples: list[list[dict[str, float | str]]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    @staticmethod
    def read() -> list[dict[str, float | str]]:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return []
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv
                [
                    exe,
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        gpus: list[dict[str, float | str]] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                try:
                    gpus.append(
                        {
                            "name": parts[0],
                            "mem_used_mb": float(parts[1]),
                            "mem_total_mb": float(parts[2]),
                            "util_pct": float(parts[3]),
                        }
                    )
                except ValueError:
                    continue
        return gpus

    def __enter__(self) -> GpuSampler:
        if self.available():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading = self.read()
            if reading:
                self.samples.append(reading)
            self._stop.wait(self.interval_s)

    def summary(self) -> list[dict[str, Any]] | None:
        if not self.samples:
            return None
        out = []
        for idx in range(len(self.samples[0])):
            rows = [s[idx] for s in self.samples if len(s) > idx]
            util = [float(r["util_pct"]) for r in rows]
            mem = [float(r["mem_used_mb"]) for r in rows]
            out.append(
                {
                    "name": rows[0]["name"],
                    "mem_total_mb": rows[0]["mem_total_mb"],
                    "mem_used_peak_mb": max(mem),
                    "util_peak_pct": max(util),
                    "util_mean_pct": round(statistics.fmean(util), 1),
                    "samples": len(rows),
                }
            )
        return out


def _get(client: httpx.Client, url: str) -> Any:
    try:
        resp = client.get(url, timeout=5.0)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        try:
            return resp.json()
        except ValueError:
            return None
    return resp.text


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name for the token/latency counters we care about."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match or not _INTERESTING.search(match.group(1)):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        totals[match.group(1)] = totals.get(match.group(1), 0.0) + value
    return totals


def _ollama_model_summary(show: dict[str, Any]) -> dict[str, Any]:
    info = show.get("model_info") or {}
    arch = info.get("general.architecture")
    ctx = info.get(f"{arch}.context_length") if arch else None
    return {
        "details": show.get("details"),
        "architecture": arch,
        "parameter_count": info.get("general.parameter_count"),
        "context_length": ctx,
        "capabilities": show.get("capabilities"),
        "parameters": show.get("parameters"),
        "modified_at": show.get("modified_at"),
    }


def server_info(client: httpx.Client, base: str, model: str | None) -> dict[str, Any]:
    """Identify the server and describe the model. Never raises for unreachable endpoints."""
    info: dict[str, Any] = {"base_url": base, "server": "unknown", "reachable": False}
    version = _get(client, f"{base}/api/version")
    if isinstance(version, dict) and "version" in version:
        info.update(server="ollama", reachable=True, version=version["version"])
        tags = _get(client, f"{base}/api/tags")
        if isinstance(tags, dict):
            info["models"] = [
                {"name": m.get("name"), "size": m.get("size"), "details": m.get("details")}
                for m in tags.get("models", [])
            ]
        ps = _get(client, f"{base}/api/ps")
        if isinstance(ps, dict):
            loaded = []
            for m in ps.get("models", []):
                size, vram = m.get("size"), m.get("size_vram")
                loaded.append(
                    {
                        "name": m.get("name"),
                        "size": size,
                        "size_vram": vram,
                        "gpu_fraction": round(vram / size, 3)
                        if size and vram is not None
                        else None,
                        "context_length": m.get("context_length"),
                        "expires_at": m.get("expires_at"),
                    }
                )
            info["loaded"] = loaded
        if model:
            try:
                resp = client.post(f"{base}/api/show", json={"model": model}, timeout=10.0)
                if resp.status_code == 200:
                    info["model"] = _ollama_model_summary(resp.json())
            except (httpx.HTTPError, ValueError):
                pass
    models = _get(client, f"{base}/v1/models")
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        info["reachable"] = True
        info["openai_models"] = [
            {k: m.get(k) for k in ("id", "owned_by", "max_model_len") if k in m}
            for m in models["data"]
            if isinstance(m, dict)
        ]
    if info["server"] == "unknown":
        vllm = _get(client, f"{base}/version")
        props = _get(client, f"{base}/props")
        if isinstance(vllm, dict) and "version" in vllm:
            info.update(server="vllm", reachable=True, version=vllm["version"])
        elif isinstance(props, dict):
            info.update(server="llama.cpp", reachable=True)
            settings = props.get("default_generation_settings") or {}
            info["model"] = {
                "n_ctx": settings.get("n_ctx") or props.get("n_ctx"),
                "model_path": props.get("model_path"),
                "total_slots": props.get("total_slots"),
            }
        elif info["reachable"]:
            info["server"] = "openai-compatible"
    metrics = _get(client, f"{base}/metrics")
    if isinstance(metrics, str):
        counters = parse_prometheus(metrics)
        if counters:
            info["counters"] = counters
    return info


def choose_model(info: dict[str, Any], requested: str | None, hints: list[str]) -> str | None:
    """Pick the model to describe/calibrate: explicit > config hint known to server > the
    single loaded/available model. Returns None if ambiguous."""
    available = [m["name"] for m in info.get("models", []) if m.get("name")]
    available += [m["id"] for m in info.get("openai_models", []) if m.get("id")]
    if requested:
        return requested
    for hint in hints:
        if hint in available:
            return hint
    loaded = [m["name"] for m in info.get("loaded", []) if m.get("name")]
    if len(loaded) == 1:
        return str(loaded[0])
    unique = sorted(set(available))
    return unique[0] if len(unique) == 1 else None


def _ollama_stream(
    client: httpx.Client, base: str, model: str, prompt: str, max_tokens: int
) -> Sample:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0, "seed": 7, "num_predict": max_tokens},
    }
    start = time.monotonic()
    ttft: float | None = None
    final: dict[str, Any] = {}
    with client.stream("POST", f"{base}/api/generate", json=body, timeout=600.0) as resp:
        if resp.status_code != 200:
            resp.read()
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if ttft is None and chunk.get("response"):
                ttft = time.monotonic() - start
            if chunk.get("done"):
                final = chunk
    total = time.monotonic() - start

    def per_s(count: Any, ns: Any) -> float | None:
        return round(count / (ns / 1e9), 2) if count and ns else None

    load_ns = final.get("load_duration")
    return Sample(
        prompt="",
        ttft_s=ttft,
        total_s=total,
        prompt_tokens=final.get("prompt_eval_count"),
        output_tokens=final.get("eval_count"),
        prompt_tps=per_s(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
        gen_tps=per_s(final.get("eval_count"), final.get("eval_duration")),
        load_s=round(load_ns / 1e9, 3) if load_ns else None,
        timing_source="server",
    )


def _openai_stream(
    client: httpx.Client, base: str, model: str, prompt: str, max_tokens: int
) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.monotonic()
    ttft: float | None = None
    usage: dict[str, Any] = {}
    with client.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=600.0) as resp:
        if resp.status_code != 200:
            resp.read()
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        for line in _sse(resp.iter_lines()):
            if line.get("usage"):
                usage = line["usage"]
            for choice in line.get("choices") or []:
                delta = choice.get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.monotonic() - start
    total = time.monotonic() - start
    out_tokens = usage.get("completion_tokens")
    in_tokens = usage.get("prompt_tokens")
    gen_window = total - ttft if ttft is not None else None
    return Sample(
        prompt="",
        ttft_s=ttft,
        total_s=total,
        prompt_tokens=in_tokens,
        output_tokens=out_tokens,
        prompt_tps=round(in_tokens / ttft, 2) if in_tokens and ttft else None,
        gen_tps=round(max(0, out_tokens - 1) / gen_window, 2)
        if out_tokens and gen_window and gen_window > 0
        else None,
        load_s=None,
        timing_source="client",
    )


def _sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def calibrate(
    client: httpx.Client, info: dict[str, Any], model: str, runs: int = 3, max_tokens: int = 200
) -> dict[str, Any]:
    """Time the fixed prompts: one warm-up (cold start), then ``runs`` measured calls each."""
    base = info["base_url"]
    call = _ollama_stream if info.get("server") == "ollama" else _openai_stream
    result: dict[str, Any] = {
        "model": model,
        "runs": runs,
        "max_tokens": max_tokens,
        "prompts": {},
        "errors": [],
    }
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    with GpuSampler() as gpu:
        try:
            cold = call(client, base, model, _SHORT_PROMPT, 16)
            result["cold_start"] = {
                "total_s": round(cold.total_s, 3),
                "load_s": cold.load_s,
                "ttft_s": cold.ttft_s,
            }
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
            result["errors"].append(f"warm-up: {exc}")
            return result
        for name, prompt in CALIBRATION_PROMPTS.items():
            samples: list[Sample] = []
            for _ in range(runs):
                try:
                    samples.append(call(client, base, model, prompt, max_tokens))
                except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
                    result["errors"].append(f"{name}: {exc}")
            result["prompts"][name] = {
                "n": len(samples),
                "timing_source": samples[0].timing_source if samples else None,
                "prompt_tokens": samples[0].prompt_tokens if samples else None,
                "output_tokens_median": _median([s.output_tokens for s in samples]),
                "ttft_s_median": _median([s.ttft_s for s in samples]),
                "total_s_median": _median([s.total_s for s in samples]),
                "prompt_tps_median": _median([s.prompt_tps for s in samples]),
                "gen_tps_median": _median([s.gen_tps for s in samples]),
                "gen_tps_min": min((s.gen_tps for s in samples if s.gen_tps), default=None),
                "gen_tps_max": max((s.gen_tps for s in samples if s.gen_tps), default=None),
            }
    result["gpu"] = gpu.summary()
    if load_before is not None:
        result["loadavg_1m"] = {
            "before": round(load_before[0], 2),
            "after": round(os.getloadavg()[0], 2),
        }
    short = result["prompts"].get("short", {})
    long = result["prompts"].get("long", {})
    result["headline"] = {
        "gen_tps": short.get("gen_tps_median"),
        "prompt_tps": long.get("prompt_tps_median"),
        "ttft_s": short.get("ttft_s_median"),
        "ttft_long_s": long.get("ttft_s_median"),
        "cold_load_s": result.get("cold_start", {}).get("load_s"),
    }
    return result


def counter_delta(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, float] | None:
    """Counter increase between two snapshots (None if either side is missing or reset)."""
    if not before or not after:
        return None
    delta = {k: round(after[k] - before[k], 3) for k in after if k in before}
    if any(v < 0 for v in delta.values()):
        return None  # server restarted between snapshots
    return delta


def probe(
    base_url: str,
    model: str | None = None,
    model_hints: list[str] | None = None,
    calibrate_runs: int = 3,
    allow_remote: bool = False,
    max_tokens: int = 200,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Full probe: server info (+ counters) and, if ``calibrate_runs`` > 0, calibration.

    Counters are read before calibrating, so an end-of-session probe does not attribute its own
    calibration traffic to the session."""
    base_url = normalize_base_url(base_url)
    host = urlparse(base_url).hostname or ""
    local = is_local_host(host)
    if not local and not allow_remote:
        raise ProbeRefusedError(
            f"{host} is not a local host; pass --allow-remote to send calibration prompts to it"
        )
    with httpx.Client(transport=transport, trust_env=not local) as client:
        info = server_info(client, base_url.rstrip("/"), model)
        chosen = choose_model(info, model, model_hints or [])
        if chosen and info.get("server") == "ollama" and "model" not in info:
            info = server_info(client, base_url.rstrip("/"), chosen)
        report: dict[str, Any] = {
            "format": PROBE_FORMAT,
            "at": utcnow(),
            "local": local,
            "model": chosen,
            "server": info,
        }
        if calibrate_runs > 0 and info.get("reachable"):
            if chosen is None:
                report["calibration_error"] = "model is ambiguous; pass --model-id"
            else:
                report["calibration"] = calibrate(client, info, chosen, calibrate_runs, max_tokens)
        if info.get("server") == "ollama" and chosen:
            ps = server_info(client, base_url.rstrip("/"), None).get("loaded")  # after load
            report["server"]["loaded"] = ps
        if not local:
            report["note"] = "remote endpoint: GPU sampling not applicable"
        report["gpu_snapshot"] = GpuSampler.read() if local else None
    return report


def headline(report: dict[str, Any] | None) -> dict[str, Any]:
    """Flat summary used by reports/dashboard (all values may be None)."""
    if not report:
        return {}
    probe_end = report.get("end") or report
    server = probe_end.get("server") or {}
    model = server.get("model") or {}
    details = model.get("details") or {}
    loaded = next(
        (m for m in server.get("loaded") or [] if m.get("name") == probe_end.get("model")),
        (server.get("loaded") or [None])[0] or {},
    )
    cal = (probe_end.get("calibration") or {}).get("headline") or {}
    gpus = (probe_end.get("calibration") or {}).get("gpu") or []
    return {
        "server": server.get("server"),
        "server_version": server.get("version"),
        "model": probe_end.get("model"),
        "quantization": details.get("quantization_level"),
        "parameter_size": details.get("parameter_size") or model.get("parameter_count"),
        "context_length": model.get("context_length")
        or (loaded or {}).get("context_length")
        or model.get("n_ctx"),
        "gpu_fraction": (loaded or {}).get("gpu_fraction"),
        "gen_tps": cal.get("gen_tps"),
        "prompt_tps": cal.get("prompt_tps"),
        "ttft_s": cal.get("ttft_s"),
        "cold_load_s": cal.get("cold_load_s"),
        "gpu": gpus[0].get("name") if gpus else None,
        "gpu_util_peak_pct": gpus[0].get("util_peak_pct") if gpus else None,
        "gpu_mem_peak_mb": gpus[0].get("mem_used_peak_mb") if gpus else None,
        "session_tokens": report.get("session_counter_delta"),
        "session_span_s": (report.get("timing") or {}).get("observed_span_s"),
    }
