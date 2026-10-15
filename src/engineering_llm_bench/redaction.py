"""Secret redaction applied to every persisted log/trace and every export."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

REDACTED = "[REDACTED]"

DEFAULT_PATTERNS: tuple[str, ...] = (
    r"sk-ant-[A-Za-z0-9_\-]{16,}",
    r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"xox[abpr]-[A-Za-z0-9\-]{10,}",
    r"(?i:bearer)\s+[A-Za-z0-9._\-]{16,}",
    r"(?i:(api[_-]?key|secret|token|password))(\s*[:=]\s*)[\"']?[^\s\"']{8,}",
)


@dataclass
class Redactor:
    """Replaces known secret values (from named env vars) and secret-shaped strings."""

    secret_env_names: Iterable[str] = ()
    extra_patterns: Iterable[str] = ()
    _values: list[str] = field(default_factory=list, init=False)
    _regex: re.Pattern[str] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._values = sorted(
            {v for n in self.secret_env_names if (v := os.environ.get(n)) and len(v) >= 6},
            key=len,
            reverse=True,
        )
        patterns = [*DEFAULT_PATTERNS, *self.extra_patterns]
        self._regex = re.compile("|".join(f"(?:{p})" for p in patterns))

    def text(self, value: str) -> str:
        for secret in self._values:
            value = value.replace(secret, REDACTED)
        if self._regex is not None:
            value = self._regex.sub(_replace, value)
        return value

    def obj(self, value: Any) -> Any:
        """Recursively redact strings inside JSON-like structures."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.obj(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.obj(v) for v in value]
        return value


def _replace(match: re.Match[str]) -> str:
    # Keep the key name for "api_key=..." style matches so logs stay readable.
    groups = [g for g in match.groups() if g is not None]
    if len(groups) >= 2 and groups[1].strip() in {":", "="}:
        return f"{groups[0]}{groups[1]}{REDACTED}"
    return REDACTED
