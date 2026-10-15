"""Atomic, content-addressed artifact files (logs, diffs, traces, final files).

Large blobs live on disk under ``<artifacts_dir>/runs/<run_id>/<attempt_id>/`` and are referenced
from SQLite by relative path + SHA-256. Writes are atomic (temp file + fsync + rename), so an
interrupted process never leaves a half-written artifact under its final name.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from engineering_llm_bench.redaction import Redactor

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}$")


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str  # relative to the artifacts root
    bytes: int
    sha256: str


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ArtifactStore:
    def __init__(self, root: Path, redactor: Redactor | None = None) -> None:
        self.root = root
        self.redactor = redactor or Redactor()
        root.mkdir(parents=True, exist_ok=True)

    def attempt_dir(self, run_id: str, attempt_id: str) -> Path:
        return self.root / "runs" / run_id / attempt_id

    def write_text(self, run_id: str, attempt_id: str, name: str, text: str) -> ArtifactRef:
        """Redact and atomically write a text artifact; return its reference."""
        if not _NAME_RE.match(name) or ".." in name.split("/"):
            raise ValueError(f"invalid artifact name {name!r}")
        data = self.redactor.text(text).encode("utf-8")
        path = self.attempt_dir(run_id, attempt_id) / name
        atomic_write_bytes(path, data)
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.root).as_posix(),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def resolve(self, relative: str) -> Path:
        """Resolve a stored relative path, refusing anything outside the artifacts root."""
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError("artifact path escapes the artifacts directory")
        return candidate
