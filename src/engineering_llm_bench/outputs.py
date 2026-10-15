"""Per-model output folders: every attempt's final code, kept separately for each model.

Layout (``outputs_dir``, default ``artifacts/model_outputs``)::

    model_outputs/
      <model-slug>/
        MODEL.json                       alias, provider, model id (first/last seen)
        runs/<run_id>/<task_id>__r<N>/
          code/...                       the attempt's final files (complete workspace)
          diff.patch                     changes against the task's starting code
          meta.json                      status, passed, score, harness, timestamps

Invisibility to other models:

* Commands run for any model (agent tools, external agent CLIs, evaluators) get a private mount
  namespace in which this whole folder - and results, task references, other attempts'
  workspaces - are covered by empty directories (see ``tools/runner.py``). This is enforced by
  the kernel on Linux (Python >= 3.12) and is the main protection.
* The folder is additionally **locked** (permissions 000) whenever the benchmark is not writing
  to it, so processes that do not run under the sandbox - e.g. an IDE agent's own terminal in a
  Copilot BYOK session - get "Permission denied". Use ``bench outputs`` to view or export it.
  (Root and the owning user can still unlock it deliberately; this is a guard, not encryption.)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from engineering_llm_bench.storage.artifacts import atomic_write_bytes
from engineering_llm_bench.storage.db import utcnow

_LOCKED_MODE = 0
_OPEN_MODE = stat.S_IRWXU


def model_slug(alias: str) -> str:
    """Filesystem-safe, readable folder name for a model label (e.g. 'strata/qwen-3.8')."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", alias).strip("-.")
    return slug[:100] or "model"


class ModelOutputs:
    """Thread-safe writer/reader for per-model output folders with lock management."""

    def __init__(self, root: Path, lock: bool = True) -> None:
        self.root = root
        self.lock_enabled = lock and sys.platform != "win32"
        self._mutex = threading.RLock()
        self._depth = 0

    # ------------------------------------------------------------------ locking
    @contextmanager
    def unlocked(self) -> Iterator[Path]:
        """Temporarily open the root for this process; re-locks when the last user leaves."""
        with self._mutex:
            if self._depth == 0:
                self.root.mkdir(parents=True, exist_ok=True)
                os.chmod(self.root, _OPEN_MODE)
            self._depth += 1
        try:
            yield self.root
        finally:
            with self._mutex:
                self._depth -= 1
                if self._depth == 0 and self.lock_enabled:
                    os.chmod(self.root, _LOCKED_MODE)

    def lock(self) -> None:
        if self.root.exists() and self.lock_enabled:
            os.chmod(self.root, _LOCKED_MODE)

    def unlock(self) -> None:
        if self.root.exists():
            os.chmod(self.root, _OPEN_MODE)

    @property
    def locked(self) -> bool:
        return self.root.exists() and stat.S_IMODE(self.root.stat().st_mode) == _LOCKED_MODE

    # ------------------------------------------------------------------ writing
    def model_dir(self, alias: str) -> Path:
        return self.root / model_slug(alias)

    def attempt_dir(self, alias: str, run_id: str, task_id: str, repeat: int) -> Path:
        return self.model_dir(alias) / "runs" / run_id / f"{task_id}__r{repeat}"

    def save_attempt(
        self,
        alias: str,
        run_id: str,
        task_id: str,
        repeat: int,
        files: Mapping[str, bytes],
        diff: str,
        meta: dict[str, Any],
        model_info: dict[str, Any] | None = None,
    ) -> Path:
        """Write one attempt's final code and metadata; returns its folder (relative to root)."""
        with self.unlocked():
            target = self.attempt_dir(alias, run_id, task_id, repeat)
            code = target / "code"
            if target.exists():
                shutil.rmtree(target)
            for rel, data in files.items():
                path = (code / rel).resolve()
                if not path.is_relative_to(code.resolve()):
                    continue  # never write outside the attempt folder
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            code.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target / "diff.patch", diff.encode("utf-8"))
            atomic_write_bytes(
                target / "meta.json",
                json.dumps({"saved_at": utcnow(), **meta}, indent=2, default=str).encode(),
            )
            self._touch_model(alias, model_info or {})
            return target.relative_to(self.root)

    def _touch_model(self, alias: str, info: dict[str, Any]) -> None:
        path = self.model_dir(alias) / "MODEL.json"
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.setdefault("alias", alias)
        data.setdefault("first_seen", utcnow())
        data["last_seen"] = utcnow()
        data.update({k: v for k, v in info.items() if v is not None})
        atomic_write_bytes(path, json.dumps(data, indent=2).encode())

    # ------------------------------------------------------------------ reading
    def list_models(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        with self.unlocked():
            out = []
            for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
                info: dict[str, Any] = {}
                if (d / "MODEL.json").is_file():
                    info = json.loads((d / "MODEL.json").read_text(encoding="utf-8"))
                runs = (
                    sorted(r.name for r in (d / "runs").iterdir()) if (d / "runs").is_dir() else []
                )
                attempts = sum(1 for _ in (d / "runs").glob("*/*")) if runs else 0
                out.append(
                    {
                        "folder": d.name,
                        "alias": info.get("alias", d.name),
                        "runs": len(runs),
                        "attempts": attempts,
                        "last_seen": info.get("last_seen"),
                    }
                )
            return out

    def export(self, alias: str, dest: Path) -> Path:
        """Copy a model's folder to ``dest`` (unlocked copy for human inspection)."""
        with self.unlocked():
            src = self.model_dir(alias)
            if not src.is_dir():
                raise FileNotFoundError(f"no outputs for model {alias!r} in {self.root}")
            target = dest / src.name
            shutil.copytree(src, target, dirs_exist_ok=True)
            return target
