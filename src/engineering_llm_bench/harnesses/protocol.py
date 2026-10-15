"""Wire formats between harnesses and models.

*Agent mode* - the model replies with exactly one JSON object per turn::

    {"action": "list_files", "path": "."}
    {"action": "read_file", "path": "src/x.py"}
    {"action": "write_file", "path": "src/x.py", "content": "..."}
    {"action": "run_command", "argv": ["pytest", "-q"]}
    {"action": "finish", "summary": "what changed, how it was tested, limitations"}

The JSON may be wrapped in a fenced code block; anything else is ignored. A plain JSON
protocol keeps the harness model-agnostic (no provider-specific tool-calling API), so local and
paid models are compared under identical conditions.

*Direct mode* - the model replies with complete files and a summary::

    <file path="src/x.py">
    ...full file content...
    </file>
    <summary>...</summary>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

ActionName = Literal["list_files", "read_file", "write_file", "run_command", "finish"]
_ACTIONS: tuple[str, ...] = ("list_files", "read_file", "write_file", "run_command", "finish")
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FILE_BLOCK = re.compile(r'<file\s+path="([^"]+)"\s*>\n?(.*?)</file>', re.DOTALL)
_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
MAX_ARGV = 64


class MalformedOutputError(ValueError):
    """The model's output does not follow the harness protocol."""


@dataclass(frozen=True)
class AgentAction:
    action: ActionName
    path: str | None = None
    content: str | None = None
    argv: tuple[str, ...] = ()
    summary: str | None = None


def _first_json_object(text: str) -> dict[str, Any]:
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    start = text.find("{")
    if start != -1:
        candidates.append(text[start:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            obj, _ = decoder.raw_decode(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise MalformedOutputError("no JSON object found; reply with exactly one JSON action object")


def parse_action(text: str) -> AgentAction:
    """Parse and validate one agent action. Raises ``MalformedOutputError`` with a hint."""
    obj = _first_json_object(text)
    name = obj.get("action")
    if name not in _ACTIONS:
        raise MalformedOutputError(f"unknown action {name!r}; expected one of {list(_ACTIONS)}")
    if name in ("read_file", "write_file", "list_files"):
        path = obj.get("path", "." if name == "list_files" else None)
        if not isinstance(path, str) or not path:
            raise MalformedOutputError(f"{name} requires a string 'path'")
        content = obj.get("content")
        if name == "write_file" and not isinstance(content, str):
            raise MalformedOutputError("write_file requires a string 'content'")
        return AgentAction(name, path=path, content=content if name == "write_file" else None)
    if name == "run_command":
        argv = obj.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or len(argv) > MAX_ARGV
            or not all(isinstance(a, str) for a in argv)
        ):
            raise MalformedOutputError("run_command requires 'argv': a non-empty list of strings")
        return AgentAction("run_command", argv=tuple(argv))
    summary = obj.get("summary", "")
    if not isinstance(summary, str):
        raise MalformedOutputError("finish requires a string 'summary'")
    return AgentAction("finish", summary=summary)


def render_action(action: dict[str, Any]) -> str:
    """Serialise an action (used by the fake provider and in tests)."""
    return json.dumps(action)


@dataclass(frozen=True)
class DirectResponse:
    files: dict[str, str]
    summary: str


def parse_direct(text: str) -> DirectResponse:
    files: dict[str, str] = {}
    for match in _FILE_BLOCK.finditer(text):
        path, content = match.group(1).strip(), match.group(2)
        files[path] = content
    summary_match = _SUMMARY.search(text)
    summary = summary_match.group(1).strip() if summary_match else ""
    if not files and not summary:
        raise MalformedOutputError('no <file path="..."> blocks or <summary> found')
    return DirectResponse(files=files, summary=summary)


def render_direct(files: dict[str, str], summary: str) -> str:
    blocks = [f'<file path="{p}">\n{c}</file>' for p, c in files.items()]
    return "\n\n".join([*blocks, f"<summary>{summary}</summary>"])
