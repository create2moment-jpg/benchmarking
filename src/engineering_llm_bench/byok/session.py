"""Copilot BYOK (and any IDE agent) sessions: prepare a self-contained workspace with a
``starting_point.md`` for the agent, then collect, validate and score what the agent produced.

Flow::

    bench byok prepare --profile standard --out ~/elb-byok/session1
        -> session1/starting_point.md        give this to the agent (Copilot agent mode)
           session1/tasks/01_<task>/TASK.md   one folder per task: fixture files + task contract
           session1/.bench/session.json       manifest (a trusted copy is kept in artifacts/)
    ... the agent edits files and writes tasks/*/RESULT.md ...
    bench byok collect ~/elb-byok/session1 --model-label "qwen-3.8"
        -> validates the folders, applies the editable changes to fresh fixture copies,
           runs the hidden evaluators and stores a run that the dashboard shows.

The session never mentions which model or serving stack (Ollama, vLLM, ...) is used: the model
label is supplied only at collection time. Hidden tests and reference solutions are never copied
into the session.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment

from engineering_llm_bench.catalog.registry import Task, TaskRegistry
from engineering_llm_bench.config.loader import ConfigError
from engineering_llm_bench.config.models import BenchmarkConfig
from engineering_llm_bench.domain import Profile
from engineering_llm_bench.importing.importer import RunImporter
from engineering_llm_bench.importing.schema import ImportedAttempt, ImportedRunMeta, ImportFile
from engineering_llm_bench.outputs import ModelOutputs
from engineering_llm_bench.storage.artifacts import ArtifactStore, atomic_write_bytes
from engineering_llm_bench.storage.db import Store, utcnow
from engineering_llm_bench.workspace.manager import (
    copy_tree,
    is_editable,
    iter_files,
    snapshot,
    snapshot_hash,
    unified_diff,
)

SESSION_FORMAT = "engineering-llm-bench/byok-session"
AGENT_FILES = {"TASK.md", "RESULT.md"}
MAX_SUBMITTED_FILE = 2_000_000
MAX_RESULT_CHARS = 50_000
_STATUS_RE = re.compile(r"^\s*\**status\**\s*:\s*\**\s*([a-z_ -]+)", re.IGNORECASE | re.MULTILINE)

#: How each logical command is offered to an IDE agent (it runs them in its own terminal).
COMMAND_HINTS = {
    "python": "`python` (run scripts / `python -c ...`)",
    "pytest": "`python -m pytest -q`",
    "ruff": "`ruff check .` (optional lint)",
    "g++": "`g++` (C++17 compiler)",
    "c++": "`c++`",
    "cmake": "`cmake -S . -B build` and `cmake --build build`",
    "ctest": "`ctest --test-dir build --output-on-failure`",
    "make": "`make`",
    "local-binaries": "run programs you compiled inside this folder, e.g. `./kf_public`",
}


@dataclass
class PreparedSession:
    session_id: str
    path: Path
    starting_point: Path
    tasks: list[str]
    trusted_manifest: Path


@dataclass
class TaskSubmission:
    task_id: str
    folder: str
    present: bool
    overlay: dict[str, bytes | None] = field(default_factory=dict)
    result_md: str | None = None
    self_reported_status: str | None = None
    rejected: list[str] = field(default_factory=list)  # non-editable text files changed
    ignored: list[str] = field(default_factory=list)  # generated/binary/oversized files
    notes: list[str] = field(default_factory=list)
    #: File modification times (epoch s) used for time estimates: first/last changed editable
    #: file and RESULT.md.
    first_change_ts: float | None = None
    last_change_ts: float | None = None
    result_ts: float | None = None

    @property
    def attempted(self) -> bool:
        return bool(self.overlay) or self.result_md is not None


@dataclass
class CollectResult:
    session_id: str
    submissions: list[TaskSubmission]
    warnings: list[str]
    run_id: str | None = None
    archived_to: Path | None = None
    session_removed: bool = False


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _jinja() -> Environment:
    return Environment(autoescape=False, keep_trailing_newline=True)  # noqa: S701 - Markdown


def _template(name: str) -> str:
    return resources.files("engineering_llm_bench.byok").joinpath(name).read_text(encoding="utf-8")


def _task_context(task: Task, folder: str) -> dict[str, Any]:
    spec = task.spec
    commands = [COMMAND_HINTS.get(c, f"`{c}`") for c in spec.allowed_commands]
    return {
        "id": spec.id,
        "version": spec.version,
        "title": spec.title,
        "domain": spec.domain.value,
        "difficulty": spec.difficulty.value,
        "folder": folder,
        "prompt": spec.prompt.strip(),
        "criteria": spec.acceptance_criteria,
        "editable": spec.editable_paths,
        "commands": commands,
        "requires": [*spec.requires.executables, *spec.requires.python_modules],
        "expected": spec.expected_artifacts,
        "estimated_min": max(1, round(spec.estimated_runtime_s / 60)),
    }


def render_task_md(task: Task, folder: str) -> str:
    return _jinja().from_string(_template("task.md.j2")).render(t=_task_context(task, folder))


def _is_inside(child: Path, parent: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def prepare_session(
    registry: TaskRegistry,
    bench: BenchmarkConfig,
    artifacts_dir: Path,
    out_dir: Path,
    profile: Profile = Profile.STANDARD,
    task_ids: list[str] | None = None,
    allow_inside_repo: bool = False,
) -> PreparedSession:
    """Create a session folder for an IDE agent. Refuses to overwrite existing content and, by
    default, refuses locations inside the benchmark repository (the agent could otherwise read
    hidden tests and reference solutions)."""
    repo_root = registry.root.resolve().parent
    if not allow_inside_repo and _is_inside(out_dir, repo_root):
        raise ConfigError(
            f"{out_dir} is inside the benchmark repository ({repo_root}); an IDE agent opened "
            "there could read hidden tests and reference solutions. Choose a folder outside the "
            "repository (or pass --allow-inside-repo for testing only)."
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ConfigError(f"{out_dir} already exists and is not empty")
    tasks = registry.select(profile, bench.profiles[profile], task_ids or None)
    if not tasks:
        raise ConfigError("no tasks selected")
    session_id = time.strftime("%Y%m%d-%H%M%S") + "-byok-" + uuid.uuid4().hex[:6]
    env = _jinja()
    entries = []
    contexts = []
    for index, task in enumerate(tasks, start=1):
        folder = f"tasks/{index:02d}_{_slug(task.id)}"
        target = out_dir / folder
        target.mkdir(parents=True)
        copy_tree(task.fixture_dir, target)
        ctx = _task_context(task, folder)
        (target / "TASK.md").write_text(render_task_md(task, folder), encoding="utf-8")
        contexts.append(ctx)
        entries.append(
            {
                "task_id": task.id,
                "version": task.spec.version,
                "task_hash": task.content_hash,
                "folder": folder,
                "fixture_hash": snapshot_hash(snapshot(task.fixture_dir)),
            }
        )
    starting_point = out_dir / "starting_point.md"
    starting_point.write_text(
        env.from_string(_template("starting_point.md.j2")).render(
            tasks=contexts,
            session_id=session_id,
            total_min=sum(c["estimated_min"] for c in contexts),
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": SESSION_FORMAT,
        "schema_version": 1,
        "session_id": session_id,
        "created_at": utcnow(),
        "suite_name": bench.suite_name,
        "suite_version": bench.suite_version,
        "profile": profile.value,
        "tasks": entries,
    }
    blob = json.dumps(manifest, indent=2).encode()
    atomic_write_bytes(out_dir / ".bench" / "session.json", blob)
    trusted = artifacts_dir / "byok_sessions" / f"{session_id}.json"
    atomic_write_bytes(trusted, blob)
    return PreparedSession(session_id, out_dir, starting_point, [t.id for t in tasks], trusted)


def _load_manifest(
    session_dir: Path, artifacts_dir: Path
) -> tuple[dict[str, Any], Path | None, list[str]]:
    warnings: list[str] = []
    local = session_dir / ".bench" / "session.json"
    if not local.is_file():
        raise ConfigError(f"{session_dir} is not a BYOK session (missing .bench/session.json)")
    try:
        data = json.loads(local.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"session manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
        raise ConfigError("session manifest has an unexpected format")
    session_id = str(data.get("session_id", ""))
    if not re.fullmatch(r"[0-9A-Za-z_.-]{6,80}", session_id):
        raise ConfigError("session manifest has an invalid session_id")
    trusted_path = artifacts_dir / "byok_sessions" / f"{session_id}.json"
    if trusted_path.is_file():
        trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
        if trusted.get("tasks") != data.get("tasks"):
            raise ConfigError(
                "the session manifest differs from the trusted copy created by `prepare`; "
                "it was modified (possibly by the agent). Restore .bench/session.json."
            )
        return trusted, trusted_path, warnings
    warnings.append(
        "trusted manifest copy not found in artifacts/ (session prepared elsewhere?); "
        "using the session's own manifest"
    )
    return data, None, warnings


def _is_text(data: bytes) -> bool:
    return b"\0" not in data[:8192]


def inspect_submission(task: Task, session_dir: Path, folder: str) -> TaskSubmission:
    """Compare the agent's task folder with the pristine fixture."""
    sub = TaskSubmission(task_id=task.id, folder=folder, present=True)
    root = session_dir / folder
    if not root.is_dir():
        sub.present = False
        sub.notes.append(f"task folder {folder} is missing")
        return sub
    baseline = snapshot(task.fixture_dir)
    current: dict[str, bytes] = {}
    mtimes: dict[str, float] = {}
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in AGENT_FILES:
            continue
        if path.stat().st_size > MAX_SUBMITTED_FILE:
            sub.ignored.append(f"{rel} (larger than {MAX_SUBMITTED_FILE} bytes)")
            continue
        current[rel] = path.read_bytes()
        mtimes[rel] = path.stat().st_mtime
    for rel in sorted(set(baseline) | set(current)):
        before, after = baseline.get(rel), current.get(rel)
        if before == after:
            continue
        if is_editable(rel, task.spec.editable_paths):
            sub.overlay[rel] = after  # None = deleted
        elif (after is not None and not _is_text(after)) or rel.endswith((".o", ".a", ".so")):
            sub.ignored.append(f"{rel} (generated/binary)")
        else:
            sub.rejected.append(rel)
    task_md = root / "TASK.md"
    expected_md = render_task_md(task, folder)
    if (
        not task_md.is_file()
        or task_md.read_text(encoding="utf-8", errors="replace") != expected_md
    ):
        sub.notes.append("TASK.md was modified or deleted (the original contract was used)")
    changed_times = [mtimes[r] for r, d in sub.overlay.items() if d is not None and r in mtimes]
    if changed_times:
        sub.first_change_ts, sub.last_change_ts = min(changed_times), max(changed_times)
    result = root / "RESULT.md"
    if result.is_file():
        sub.result_ts = result.stat().st_mtime
        text = result.read_text(encoding="utf-8", errors="replace")[:MAX_RESULT_CHARS]
        sub.result_md = text
        match = _STATUS_RE.search(text)
        if match:
            sub.self_reported_status = match.group(1).strip().lower()
    if sub.rejected:
        sub.notes.append(
            "changes to non-editable files were IGNORED (not evaluated): " + ", ".join(sub.rejected)
        )
    if not sub.attempted:
        sub.notes.append("no changes and no RESULT.md: treated as not attempted")
    return sub


def collect_session(
    session_dir: Path,
    registry: TaskRegistry,
    bench: BenchmarkConfig,
    store: Store | None,
    artifacts: ArtifactStore | None,
    artifacts_dir: Path,
    *,
    model_label: str,
    harness_label: str = "copilot-byok",
    harness_version: str | None = None,
    provider_label: str | None = None,
    remote: bool | None = None,
    session_minutes: float | None = None,
    notes: str | None = None,
    only_attempted: bool = False,
    dry_run: bool = False,
    force: bool = False,
    outputs: ModelOutputs | None = None,
    keep_session: bool = False,
) -> CollectResult:
    """Validate a session folder and (unless ``dry_run``) evaluate and persist it as a run."""
    manifest, trusted_path, warnings = _load_manifest(session_dir, artifacts_dir)
    session_id = manifest["session_id"]
    previous = manifest.get("collected_runs", [])
    if previous and not dry_run and not force:
        raise ConfigError(
            f"session {session_id} was already collected as run(s) "
            f"{[p['run_id'] for p in previous]}; collecting again would double-count it. "
            "Pass --force to collect anyway."
        )
    if outputs is None:
        outputs = ModelOutputs(artifacts_dir / "model_outputs", bench.isolation.lock_outputs)
    with ExitStack() as stack:
        source = session_dir
        if not (session_dir / "tasks").is_dir():
            source = stack.enter_context(_restored_session(outputs, previous, session_id))
        return _collect(
            source,
            session_dir,
            manifest,
            trusted_path,
            warnings,
            registry,
            bench,
            store,
            artifacts,
            outputs,
            dict(
                model_label=model_label,
                harness_label=harness_label,
                harness_version=harness_version,
                provider_label=provider_label,
                remote=remote,
                session_minutes=session_minutes,
                notes=notes,
                only_attempted=only_attempted,
                dry_run=dry_run,
                keep_session=keep_session,
            ),
        )
    raise AssertionError("unreachable")  # pragma: no cover


@contextmanager
def _restored_session(
    outputs: ModelOutputs, previous: list[dict[str, Any]], session_id: str
) -> Iterator[Path]:
    """A temporary copy of an already collected (and removed) session, from its archive."""
    with outputs.unlocked():
        archives = [
            outputs.model_dir(p["model_label"]) / "byok_sessions" / session_id
            for p in previous
            if p.get("model_label")
        ]
        found = next((a for a in archives if (a / "tasks").is_dir()), None)
        if found is None:
            raise ConfigError(
                f"session {session_id} has no tasks/ folder and no archived copy was found in "
                f"{outputs.root}"
            )
        tmp = Path(tempfile.mkdtemp(prefix="elb-recollect-"))
        shutil.copytree(found, tmp / "session", symlinks=True)
    try:
        yield tmp / "session"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _collect(
    source: Path,
    session_dir: Path,
    manifest: dict[str, Any],
    trusted_path: Path | None,
    warnings: list[str],
    registry: TaskRegistry,
    bench: BenchmarkConfig,
    store: Store | None,
    artifacts: ArtifactStore | None,
    outputs: ModelOutputs,
    opts: dict[str, Any],
) -> CollectResult:
    session_id = manifest["session_id"]
    model_label: str = opts["model_label"]
    only_attempted: bool = opts["only_attempted"]
    submissions: list[TaskSubmission] = []
    for entry in manifest["tasks"]:
        task_id = entry["task_id"]
        if task_id not in registry.tasks:
            raise ConfigError(f"session task {task_id!r} is not in the task catalogue")
        task = registry.get(task_id)
        if entry.get("task_hash") != task.content_hash:
            warnings.append(
                f"{task_id}: task changed since the session was prepared "
                f"(v{entry.get('version')} -> v{task.spec.version}); results are not comparable"
            )
        submissions.append(inspect_submission(task, source, entry["folder"]))
    result = CollectResult(session_id, submissions, warnings)
    if opts["dry_run"]:
        return result
    if store is None or artifacts is None:
        raise ConfigError("store and artifacts are required unless dry_run")

    attempts, overlays = _build_attempts(submissions, registry, only_attempted)
    run_notes = [f"BYOK session {session_id}"]
    if opts["session_minutes"] is not None:
        run_notes.append(f"whole-session wall time reported by user: {opts['session_minutes']} min")
    if opts["notes"]:
        run_notes.append(opts["notes"])
    doc = ImportFile(
        run=ImportedRunMeta(
            model_label=model_label,
            provider_label=opts["provider_label"],
            harness=opts["harness_label"],
            harness_version=opts["harness_version"],
            remote=opts["remote"],
            notes=" | ".join(run_notes),
        ),
        attempts=attempts,
    )
    importer = RunImporter(bench, registry, store, artifacts, outputs)
    imported = importer.import_document(
        doc,
        source,
        f"byok-session:{session_id}",
        overlays=overlays,
        run_config={"byok_session": session_id, "only_attempted": only_attempted},
    )
    result.run_id = imported.run_id
    result.warnings.extend(imported.warnings)
    result.archived_to = archive_session(source, importer.outputs, model_label, session_id)
    if not opts["keep_session"]:
        remove_session_workspace(session_dir, session_id, model_label)
        result.session_removed = True
    if trusted_path is not None:
        manifest.setdefault("collected_runs", []).append(
            {"run_id": imported.run_id, "model_label": model_label, "at": utcnow()}
        )
        atomic_write_bytes(trusted_path, json.dumps(manifest, indent=2).encode())
    return result


def _build_attempts(
    submissions: list[TaskSubmission], registry: TaskRegistry, only_attempted: bool
) -> tuple[list[ImportedAttempt], dict[str, dict[str, bytes | None]]]:
    """Turn validated submissions into import attempts (+ file overlays to evaluate)."""
    attempts: list[ImportedAttempt] = []
    overlays: dict[str, dict[str, bytes | None]] = {}
    for sub in submissions:
        task = registry.get(sub.task_id)
        evaluate_it = sub.attempted or not only_attempted
        patch: str | None = None
        if evaluate_it:
            baseline = snapshot(task.fixture_dir)
            after = dict(baseline)
            for rel, data in sub.overlay.items():
                if data is None:
                    after.pop(rel, None)
                else:
                    after[rel] = data
            patch = unified_diff(baseline, after) or None
            overlays[sub.task_id] = sub.overlay
        human = list(sub.notes)
        if sub.self_reported_status:
            human.append(f"agent self-reported status: {sub.self_reported_status} (not trusted)")
        if sub.ignored:
            human.append("ignored files: " + ", ".join(sub.ignored[:20]))
        attempts.append(
            ImportedAttempt(
                task_id=sub.task_id,
                task_version=task.spec.version,
                patch=patch,
                summary=sub.result_md,
                human_notes="; ".join(human) or None,
            )
        )
    return attempts, overlays


def agent_prompt() -> str:
    """The prompt to send to the IDE agent together with starting_point.md (one paragraph,
    same text as instructions/command.md)."""
    text = resources.files("engineering_llm_bench.byok").joinpath("command.md").read_text("utf-8")
    return " ".join(text.split())


def trusted_manifest_path(artifacts_dir: Path, session_id: str) -> Path:
    return artifacts_dir / "byok_sessions" / f"{session_id}.json"


def read_session_id(session_dir: Path) -> str:
    data = json.loads((session_dir / ".bench" / "session.json").read_text(encoding="utf-8"))
    session_id = str(data.get("session_id", ""))
    if not re.fullmatch(r"[0-9A-Za-z_.-]{6,80}", session_id):
        raise ConfigError("session manifest has an invalid session_id")
    return session_id


def update_trusted_manifest(artifacts_dir: Path, session_id: str, key: str, value: Any) -> None:
    path = trusted_manifest_path(artifacts_dir, session_id)
    if not path.is_file():
        raise ConfigError(f"trusted manifest for {session_id} not found in {path.parent}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    atomic_write_bytes(path, json.dumps(data, indent=2).encode())


def read_trusted_manifest(artifacts_dir: Path, session_id: str) -> dict[str, Any] | None:
    path = trusted_manifest_path(artifacts_dir, session_id)
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def estimate_timing(submissions: list[TaskSubmission], started_at: float | None) -> dict[str, Any]:
    """Per-task active time estimated from file modification times (tasks are done in order).

    A task's end is its RESULT.md (else its last edit); its start is the previous task's end, or
    the recorded session start / its own first edit for the first task. These are ESTIMATES and
    are reported separately from measured wall times.
    """
    tasks = []
    prev_end = started_at
    ends: list[float] = []
    for sub in submissions:
        end = sub.result_ts or sub.last_change_ts
        start = prev_end if prev_end is not None else sub.first_change_ts
        estimate = round(end - start, 1) if end and start and end >= start else None
        tasks.append({"task_id": sub.task_id, "estimated_s": estimate})
        if end:
            ends.append(end)
            prev_end = end
    first = started_at or min(
        (s.first_change_ts for s in submissions if s.first_change_ts), default=None
    )
    span = round(max(ends) - first, 1) if ends and first and max(ends) >= first else None
    return {
        "method": "file modification times (estimate)",
        "session_started_at_epoch": started_at,
        "observed_span_s": span,
        "tasks": tasks,
    }


def archive_session(
    session_dir: Path, outputs: ModelOutputs, model_label: str, session_id: str
) -> Path:
    """Copy the raw session (agent's files, RESULT.md, rejected edits) into the model's private
    output folder. Symlinks are copied as links, never followed."""
    with outputs.unlocked():
        dest = outputs.model_dir(model_label) / "byok_sessions" / session_id
        shutil.copytree(
            session_dir,
            dest,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "build", "*.o"),
            dirs_exist_ok=True,
        )
        return dest


def remove_session_workspace(session_dir: Path, session_id: str, model_label: str) -> None:
    """Delete the agent-visible working copy (already archived) so a later model's agent cannot
    find this model's code; leave a short note behind."""
    tasks = session_dir / "tasks"
    if tasks.is_dir():
        shutil.rmtree(tasks)
    (session_dir / "starting_point.md").unlink(missing_ok=True)
    (session_dir / "COLLECTED.md").write_text(
        f"Session {session_id} was collected for the benchmark. The code was moved to the "
        "benchmark's private per-model output folder (`bench outputs list`).\n",
        encoding="utf-8",
    )


def uncollected_sibling_sessions(target: Path) -> list[Path]:
    """Other session folders next to ``target`` that still contain an agent's working copy."""
    parent = target.expanduser().resolve().parent
    if not parent.is_dir():
        return []
    return sorted(
        p
        for p in parent.iterdir()
        if p.is_dir()
        and p != target
        and (p / ".bench" / "session.json").is_file()
        and (p / "tasks").is_dir()
    )
