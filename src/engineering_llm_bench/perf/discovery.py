"""Find model-server endpoints in an IDE / Copilot BYOK configuration file.

VS Code settings and similar files are JSON with comments and trailing commas (JSONC), and the
exact keys differ between Copilot versions and providers. Instead of hard-coding key names, this
walks the document and reports every http(s) URL together with nearby model identifiers, plus an
implicit default Ollama endpoint when an "ollama" key is configured without a URL. Pass
``--endpoint`` to skip discovery entirely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_OLLAMA = "http://localhost:11434"
MAX_CONFIG_BYTES = 5_000_000
_URL_RE = re.compile(r"^https?://[^\s\"']+$")
_MODEL_KEYS = ("model", "model_id", "modelId", "id", "name")


@dataclass
class EndpointCandidate:
    url: str
    json_path: str
    model_ids: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return normalize_base_url(self.url)


def normalize_base_url(url: str) -> str:
    """Server root for probing: strips API suffixes such as /v1, /api, /v1/chat/completions."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"not an http(s) URL: {url!r}")
    path = parsed.path.rstrip("/")
    path = re.sub(r"/(v1|api)(/.*)?$", "", path)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}{path}"


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments (outside strings) and trailing commas."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(ch)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def load_jsonc(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"config file too large: {path}")
    return json.loads(strip_jsonc(path.read_text(encoding="utf-8")))


_GENERIC_KEYS = {"settings", "models", "endpoints", "providers", "config", "byok"}


def _model_ids_near(node: Any, container_key: str | None) -> list[str]:
    """Model ids stored next to a URL (``{"id": ..., "url": ...}``) or used as the key of the
    object holding the URL (``{"qwen3:8b": {"url": ...}}``)."""
    ids: list[str] = []
    if isinstance(node, dict):
        for k in _MODEL_KEYS:
            v = node.get(k)
            if isinstance(v, str) and v and not _URL_RE.match(v):
                ids.append(v)
    if (
        container_key
        and container_key.lower() not in _GENERIC_KEYS
        and not container_key.startswith(("github.", "http"))
    ):
        ids.append(container_key)
    return ids


def discover_endpoints(config: Any) -> list[EndpointCandidate]:
    """Every http(s) URL in ``config`` with model ids found next to it (deduplicated)."""
    found: dict[str, EndpointCandidate] = {}
    mentions_ollama = False

    def walk(node: Any, path: str, container_key: str | None) -> None:
        nonlocal mentions_ollama
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k)
                if "ollama" in key.lower() or (isinstance(v, str) and v.lower() == "ollama"):
                    mentions_ollama = True
                child_path = f"{path}.{key}" if path else key
                if isinstance(v, str) and _URL_RE.match(v.strip()):
                    _record(found, v.strip(), child_path, _model_ids_near(node, container_key))
                else:
                    walk(v, child_path, key)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str) and _URL_RE.match(v.strip()):
                    _record(found, v.strip(), f"{path}[{i}]", [])
                else:
                    walk(v, f"{path}[{i}]", None)

    walk(config, "", None)
    ollama_url_given = any(
        "ollama" in c.json_path.lower() or urlparse(c.url).port == 11434 for c in found.values()
    )
    if mentions_ollama and not ollama_url_given:
        found.setdefault(
            DEFAULT_OLLAMA, EndpointCandidate(DEFAULT_OLLAMA, "(implicit Ollama default)")
        )
    return list(found.values())


def _record(found: dict[str, EndpointCandidate], url: str, path: str, ids: list[str]) -> None:
    try:
        base = normalize_base_url(url)
    except ValueError:
        return
    cand = found.setdefault(base, EndpointCandidate(url, path))
    for mid in ids:
        if mid not in cand.model_ids:
            cand.model_ids.append(mid)
