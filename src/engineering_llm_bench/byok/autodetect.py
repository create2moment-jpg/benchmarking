"""Zero-input helpers for the one-command BYOK workflow (``setup.sh``, ``finish.sh``):
find the model server Copilot uses, guess which model ran, and remember the current session."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import is_local_host
from engineering_llm_bench.perf.capture import discover_from_file, vscode_settings_candidates
from engineering_llm_bench.perf.discovery import normalize_base_url
from engineering_llm_bench.storage.artifacts import atomic_write_bytes

OLLAMA_DEFAULT = "http://127.0.0.1:11434"
CURRENT_SESSION_FILE = "byok_current_session.json"
_TIMEOUT = 3.0


@dataclass
class FoundEndpoint:
    base_url: str
    source: str  # settings file it came from, or "default Ollama port"
    model_ids: list[str]


def _get_json(url: str) -> Any:
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None


def is_ollama(base_url: str) -> bool:
    return isinstance(_get_json(f"{base_url}/api/version"), dict)


def find_endpoint(settings_files: list[Path] | None = None) -> FoundEndpoint | None:
    """The model server configured for Copilot BYOK in the VS Code user settings; if none (or
    several) is configured, a local Ollama on its default port when it answers."""
    files = vscode_settings_candidates() if settings_files is None else settings_files
    found: dict[str, FoundEndpoint] = {}
    for path in files:
        try:
            candidates = discover_from_file(path)
        except ConfigError:
            continue
        for c in candidates:
            try:
                base = c.base_url
            except ValueError:
                continue
            entry = found.setdefault(base, FoundEndpoint(base, str(path), []))
            entry.model_ids.extend(m for m in c.model_ids if m not in entry.model_ids)
    if len(found) == 1:
        return next(iter(found.values()))
    if is_ollama(OLLAMA_DEFAULT):
        return FoundEndpoint(OLLAMA_DEFAULT, "default Ollama port", [])
    return None


def available_models(base_url: str) -> list[str]:
    """Models the server offers (Ollama ``/api/tags`` or OpenAI-style ``/v1/models``)."""
    tags = _get_json(f"{base_url}/api/tags")
    if isinstance(tags, dict) and isinstance(tags.get("models"), list):
        return [str(m.get("name")) for m in tags["models"] if isinstance(m, dict) and m.get("name")]
    for path in ("/v1/models", "/models"):
        data = _get_json(f"{base_url.removesuffix('/v1')}{path}")
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [str(m.get("id")) for m in data["data"] if isinstance(m, dict) and m.get("id")]
    return []


def loaded_models(base_url: str) -> list[str]:
    """Models currently loaded in memory (Ollama ``/api/ps``); empty when unknown."""
    data = _get_json(f"{base_url}/api/ps")
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [str(m.get("name")) for m in data["models"] if isinstance(m, dict) and m.get("name")]
    return []


def guess_model(endpoint: FoundEndpoint | None) -> tuple[str | None, list[str]]:
    """(best guess of the model that just ran, choices to offer). A guess is returned only when
    it is unambiguous: exactly one model loaded right now, or exactly one model offered."""
    if endpoint is None:
        return None, []
    loaded = loaded_models(endpoint.base_url)
    if len(loaded) == 1:
        return loaded[0], loaded
    offered = available_models(endpoint.base_url) or endpoint.model_ids
    if len(offered) == 1:
        return offered[0], offered
    return None, loaded or offered


def endpoint_is_local(base_url: str) -> bool:
    from urllib.parse import urlparse  # noqa: PLC0415

    return is_local_host(urlparse(normalize_base_url(base_url)).hostname or "")


# ------------------------------------------------------------------------------ current session
def write_current_session(artifacts_dir: Path, data: dict[str, Any]) -> Path:
    path = artifacts_dir / CURRENT_SESSION_FILE
    atomic_write_bytes(path, json.dumps(data, indent=2).encode())
    return path


def read_current_session(artifacts_dir: Path) -> dict[str, Any] | None:
    path = artifacts_dir / CURRENT_SESSION_FILE
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def clear_current_session(artifacts_dir: Path) -> None:
    (artifacts_dir / CURRENT_SESSION_FILE).unlink(missing_ok=True)


# ------------------------------------------------------------------------------ desktop helpers
_CLIPBOARD_COMMANDS = (
    ["wl-copy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
    ["pbcopy"],
    ["clip.exe"],
)


def copy_to_clipboard(text: str) -> bool:
    for argv in _CLIPBOARD_COMMANDS:
        if shutil.which(argv[0]) is None:
            continue
        try:
            subprocess.run(argv, input=text.encode(), check=True, timeout=5)  # noqa: S603
        except (OSError, subprocess.SubprocessError):
            continue
        return True
    return False


def devcontainer_uri(folder: Path, workspace: str = "/workspace") -> str:
    """URI that makes VS Code open ``folder`` directly inside its Dev Container."""
    return f"vscode-remote://dev-container+{str(folder).encode().hex()}{workspace}"


def open_in_vscode(folder: Path, in_container: bool = False) -> bool:
    """Open ``folder`` in a new VS Code window (``code -n``), inside its Dev Container if asked
    (needs the Dev Containers extension; otherwise VS Code offers "Reopen in Container")."""
    target = ["--folder-uri", devcontainer_uri(folder)] if in_container else [str(folder)]
    for exe in ("code", "code-insiders", "codium"):
        path = shutil.which(exe)
        if path is None:
            continue
        try:
            subprocess.Popen(  # noqa: S603
                [path, "-n", *target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            continue
        return True
    return False
