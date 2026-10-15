"""Glue between endpoint discovery, probing and stored runs/sessions."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.perf.discovery import (
    EndpointCandidate,
    discover_endpoints,
    load_jsonc,
    normalize_base_url,
)
from engineering_llm_bench.perf.probe import ProbeRefusedError, counter_delta, probe


def vscode_settings_candidates() -> list[Path]:
    """Common VS Code user-settings locations (Copilot BYOK settings usually live there)."""
    home = Path.home()
    roots = [
        home / ".config",
        home / "Library" / "Application Support",
        Path(os.environ["APPDATA"]) if os.environ.get("APPDATA") else None,
    ]
    out = []
    for root in roots:
        if root is None:
            continue
        for product in ("Code", "Code - Insiders", "VSCodium"):
            path = root / product / "User" / "settings.json"
            if path.is_file():
                out.append(path)
    return out


def discover_from_file(path: Path) -> list[EndpointCandidate]:
    try:
        return discover_endpoints(load_jsonc(path))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read BYOK config {path}: {exc}") from exc


def resolve_endpoint(
    endpoint: str | None, byok_config: Path | None, model_id: str | None
) -> tuple[str, list[str]]:
    """Return (server base URL, model-id hints) from an explicit URL or a BYOK config file."""
    if endpoint:
        try:
            return normalize_base_url(endpoint), [model_id] if model_id else []
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if byok_config is None:
        raise ConfigError("pass --endpoint URL or --byok-config FILE to locate the model server")
    candidates = discover_from_file(byok_config)
    if not candidates:
        raise ConfigError(f"no http(s) endpoint found in {byok_config}; pass --endpoint")
    if model_id:
        matching = [c for c in candidates if model_id in c.model_ids]
        if len(matching) == 1:
            return matching[0].base_url, [model_id, *matching[0].model_ids]
    if len(candidates) > 1:
        listing = "; ".join(
            f"{c.base_url} (at {c.json_path}, models {c.model_ids})" for c in candidates
        )
        raise ConfigError(
            f"several endpoints in {byok_config}: {listing}. Choose one with --endpoint "
            "(and --model-id)"
        )
    hints = ([model_id] if model_id else []) + candidates[0].model_ids
    return candidates[0].base_url, hints


def snapshot(
    base_url: str,
    model_id: str | None,
    hints: list[str],
    calibrate_runs: int,
    allow_remote: bool,
    max_tokens: int = 200,
) -> dict[str, Any]:
    """Probe without raising: unreachable servers and refusals become an ``error`` entry."""
    try:
        return probe(base_url, model_id, hints, calibrate_runs, allow_remote, max_tokens)
    except ProbeRefusedError as exc:
        return {"error": str(exc), "base_url": base_url}
    except Exception as exc:  # network edge cases: never break a benchmark run
        return {"error": f"{type(exc).__name__}: {exc}", "base_url": base_url}


def assemble(
    base_url: str,
    start: dict[str, Any] | None,
    end: dict[str, Any] | None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run-level performance record stored in ``runs.perf_json``."""
    before = ((start or {}).get("server") or {}).get("counters")
    after = ((end or {}).get("server") or {}).get("counters")
    return {
        "format": "engineering-llm-bench/run-perf",
        "endpoint": base_url,
        "model": (end or {}).get("model") or (start or {}).get("model"),
        "start": start,
        "end": end,
        "session_counter_delta": counter_delta(before, after),
        "timing": timing,
        "recorded_at_epoch": time.time(),
    }
