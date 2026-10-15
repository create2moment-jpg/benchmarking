"""Task discovery, validation, content hashing, availability and profile selection."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from engineering_llm_bench.catalog.schema import TaskSpec
from engineering_llm_bench.config.loader import ConfigError, parse_model, read_yaml
from engineering_llm_bench.config.models import ProfileConfig
from engineering_llm_bench.domain import Profile

_IGNORED_PARTS = {"__pycache__", ".pytest_cache", "build", ".ruff_cache", ".mypy_cache"}


@dataclass(frozen=True)
class Task:
    """A loaded task: validated spec plus its on-disk location and content hash."""

    spec: TaskSpec
    root: Path
    content_hash: str

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def fixture_dir(self) -> Path:
        return self.root / "fixture"

    @property
    def evaluator_dir(self) -> Path:
        return self.root / "evaluator"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def broken_dir(self) -> Path:
        return self.root / "broken"


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not (_IGNORED_PARTS & set(p.relative_to(root).parts))
    )


def hash_tree(root: Path) -> str:
    """Stable SHA-256 over relative paths and file contents (ignores caches/build outputs)."""
    digest = hashlib.sha256()
    for path in _iter_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_task(task_yaml: Path) -> Task:
    """Load and validate a single ``task.yaml``; raise ``ConfigError`` with context."""
    spec = parse_model(TaskSpec, read_yaml(task_yaml), str(task_yaml))
    root = task_yaml.parent
    if not (root / "fixture").is_dir():
        raise ConfigError(f"{task_yaml}: missing fixture/ directory")
    return Task(spec=spec, root=root, content_hash=hash_tree(root))


def check_availability(task: Task) -> Availability:
    """Report whether the task's required executables and Python modules are present."""
    missing: list[str] = []
    for exe in task.spec.requires.executables:
        if shutil.which(exe) is None:
            missing.append(f"executable '{exe}'")
    for module in task.spec.requires.python_modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(f"python module '{module}'")
    if missing:
        return Availability(False, "missing " + ", ".join(missing))
    return Availability(True)


@dataclass
class TaskRegistry:
    """All tasks under a tasks directory, keyed by task id."""

    root: Path
    tasks: dict[str, Task] = field(default_factory=dict)

    @classmethod
    def discover(cls, root: Path) -> TaskRegistry:
        if not root.is_dir():
            raise ConfigError(f"tasks directory not found: {root}")
        registry = cls(root=root)
        errors: list[str] = []
        for task_yaml in sorted(root.rglob("task.yaml")):
            try:
                task = load_task(task_yaml)
            except ConfigError as exc:
                errors.append(str(exc))
                continue
            if task.id in registry.tasks:
                errors.append(f"duplicate task id {task.id!r} at {task_yaml}")
                continue
            registry.tasks[task.id] = task
        if errors:
            raise ConfigError("invalid task catalogue:\n" + "\n".join(errors))
        return registry

    def get(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise ConfigError(
                f"unknown task id {task_id!r}; run `bench list-tasks` to see available tasks"
            ) from None

    def unknown_ids(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self.tasks]

    def select(
        self,
        profile: Profile,
        profile_cfg: ProfileConfig | None = None,
        task_ids: list[str] | None = None,
        tags: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> list[Task]:
        """Select tasks for a profile, optionally narrowed by explicit ids, tags or domains.

        Explicit ``task_ids`` override profile membership (so a single task can be run under any
        profile's budget). Unknown ids raise ``ConfigError``.
        """
        if task_ids:
            unknown = self.unknown_ids(task_ids)
            if unknown:
                raise ConfigError(f"unknown task ids: {unknown}")
            selected = [self.tasks[i] for i in task_ids]
        else:
            selected = [
                t
                for t in self.tasks.values()
                if profile == Profile.EXTENDED or profile in t.spec.profiles
            ]
            if profile_cfg is not None:
                unknown = self.unknown_ids(profile_cfg.include_tasks + profile_cfg.exclude_tasks)
                if unknown:
                    raise ConfigError(f"profile {profile} references unknown task ids: {unknown}")
                ids = {t.id for t in selected} | set(profile_cfg.include_tasks)
                ids -= set(profile_cfg.exclude_tasks)
                selected = [self.tasks[i] for i in ids]
        if tags:
            selected = [t for t in selected if set(tags) & set(t.spec.tags)]
        if domains:
            selected = [t for t in selected if t.spec.domain.value in domains]
        return sorted(selected, key=lambda t: t.id)
