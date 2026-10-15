"""Disposable per-attempt workspaces, safe path resolution, snapshots and diffs.

Each attempt gets a fresh temporary directory::

    <tmp>/elb-XXXX/
        ws/        copy of the task fixture; the only directory the model may edit
        scratch/   HOME/TMPDIR for commands (excluded from diffs)
        eval/      fresh copy of the final ws/ plus hidden evaluator files in eval/_hidden/

Workspaces live outside the repository so attempts cannot see other runs, reference solutions
or hidden tests through relative paths. They are deleted after evidence has been persisted
unless ``isolation.keep_workspaces`` is set.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from engineering_llm_bench.catalog.registry import Task

HIDDEN_DIRNAME = "_hidden"
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", ".git"}
_TEXT_LIMIT = 1_000_000


class PathEscapeError(ValueError):
    """A requested path resolves outside its root (traversal or symlink escape)."""


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` and refuse anything that escapes it."""
    if not relative or "\x00" in relative:
        raise PathEscapeError("empty or invalid path")
    rel = PurePosixPath(relative.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise PathEscapeError(f"path {relative!r} must be relative and must not contain '..'")
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise PathEscapeError(f"path {relative!r} escapes the workspace")
    return candidate


def is_editable(relative: str, patterns: list[str]) -> bool:
    rel = PurePosixPath(relative).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def iter_files(root: Path) -> list[Path]:
    """Regular files under ``root`` (symlinks and cache/build directories are skipped)."""
    out: list[Path] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if _SKIP_DIRS & set(rel_parts) or path.is_symlink() or not path.is_file():
            continue
        out.append(path)
    return out


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in iter_files(root)}


def snapshot_hash(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode() + b"\0" + files[rel] + b"\0")
    return digest.hexdigest()


def _lines(data: bytes) -> list[str] | None:
    if b"\0" in data[:8192] or len(data) > _TEXT_LIMIT:
        return None
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def unified_diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    """Git-style unified diff between two snapshots (binary files summarised)."""
    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        old, new = before.get(rel), after.get(rel)
        if old == new:
            continue
        old_lines = _lines(old) if old is not None else []
        new_lines = _lines(new) if new is not None else []
        if old_lines is None or new_lines is None:
            chunks.append(f"Binary files a/{rel} and b/{rel} differ\n")
            continue
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}" if old is not None else "/dev/null",
            tofile=f"b/{rel}" if new is not None else "/dev/null",
        )
        text = "".join(line if line.endswith("\n") else line + "\n" for line in diff)
        chunks.append(f"diff --git a/{rel} b/{rel}\n{text}")
    return "".join(chunks)


def copy_tree(src: Path, dst: Path) -> None:
    """Copy regular files from ``src`` into ``dst`` (overlaying existing files)."""
    for path in iter_files(src):
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


@dataclass
class Workspace:
    base: Path
    task: Task
    baseline: dict[str, bytes]
    keep: bool = False

    @property
    def root(self) -> Path:
        return self.base / "ws"

    @property
    def scratch(self) -> Path:
        return self.base / "scratch"

    @property
    def baseline_hash(self) -> str:
        return snapshot_hash(self.baseline)

    def current(self) -> dict[str, bytes]:
        return snapshot(self.root)

    def diff(self) -> str:
        return unified_diff(self.baseline, self.current())

    def changed_files(self) -> list[str]:
        after = self.current()
        return sorted(
            rel
            for rel in set(self.baseline) | set(after)
            if self.baseline.get(rel) != after.get(rel)
        )

    def apply_overlay(self, overlay_dir: Path) -> None:
        copy_tree(overlay_dir, self.root)

    def make_eval_copy(self, name: str = "eval") -> Path:
        """Fresh evaluation directory = final workspace + hidden evaluator files."""
        dest = self.base / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        copy_tree(self.root, dest)
        hidden = dest / HIDDEN_DIRNAME
        if hidden.exists():  # the model must not be able to pre-seed hidden files
            shutil.rmtree(hidden)
        if self.task.evaluator_dir.is_dir():
            hidden.mkdir()
            copy_tree(self.task.evaluator_dir, hidden)
        return dest

    def make_baseline_copy(self, name: str = "baseline") -> Path:
        """Directory containing the original fixture (used by regression-test checks)."""
        dest = self.base / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for rel, data in self.baseline.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return dest

    def cleanup(self) -> None:
        if not self.keep:
            shutil.rmtree(self.base, ignore_errors=True)


def workspaces_root() -> Path:
    """Private parent of all attempt workspaces (mode 700). Commands run for one attempt see
    every sibling attempt directory as empty (mount-namespace hiding)."""
    root = (
        Path(tempfile.gettempdir())
        / f"elb-workspaces-{os.getuid() if hasattr(os, 'getuid') else 0}"
    )
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def prepare_workspace(task: Task, keep: bool = False) -> Workspace:
    """Create a fresh workspace from the task fixture. Nothing from previous runs leaks in."""
    base = Path(tempfile.mkdtemp(prefix="elb-", dir=workspaces_root()))
    (base / "ws").mkdir()
    (base / "scratch").mkdir()
    copy_tree(task.fixture_dir, base / "ws")
    return Workspace(base=base, task=task, baseline=snapshot(base / "ws"), keep=keep)
