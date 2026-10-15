"""Environment fingerprinting: what ran the benchmark, recorded with every run."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any

from engineering_llm_bench import __version__

TOOLS = ("g++", "c++", "clang++", "cmake", "make", "java", "git")
PY_MODULES = ("pytest", "ruff", "numpy", "duckdb", "torch", "pyspark")


def tool_version(executable: str, timeout_s: float = 5.0) -> str | None:
    """First line of ``<tool> --version`` or None if unavailable. Never raises."""
    path = shutil.which(executable)
    if path is None:
        return None
    try:
        proc = subprocess.run(  # structured argv, no shell
            [path, "--version"], capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (proc.stdout or proc.stderr).strip().splitlines()
    return text[0][:200] if text else "unknown"


def module_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect a JSON-serialisable description of the benchmark host (no secrets, no paths
    beyond the interpreter location)."""
    return {
        "bench_version": __version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tools": {t: tool_version(t) for t in TOOLS},
        "python_modules": {m: module_version(m) for m in PY_MODULES},
    }


def fingerprint(data: Any) -> str:
    """Short stable hash of a JSON-serialisable value (sorted keys)."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def environment_fingerprint(env: dict[str, Any]) -> str:
    """Fingerprint of the parts of the environment that can change evaluation outcomes."""
    relevant = {
        "python": ".".join(str(env.get("python", "")).split(".")[:2]),
        "tools": env.get("tools", {}),
        "python_modules": env.get("python_modules", {}),
        "machine": env.get("machine"),
    }
    return fingerprint(relevant)
