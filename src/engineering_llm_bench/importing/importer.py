"""Import manually captured runs and (optionally) evaluate their patches independently."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engineering_llm_bench.catalog.registry import Task, TaskRegistry
from engineering_llm_bench.config.loader import ConfigError, parse_model
from engineering_llm_bench.config.models import BenchmarkConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import AttemptStatus, ExecutionMode, FailureClass, RunStatus
from engineering_llm_bench.environment import (
    collect_environment,
    environment_fingerprint,
    fingerprint,
)
from engineering_llm_bench.evaluation.evaluator import evaluate
from engineering_llm_bench.harnesses.base import HarnessResult
from engineering_llm_bench.importing.schema import ImportedAttempt, ImportFile
from engineering_llm_bench.orchestration.records import artifact_rows, evaluation_rows
from engineering_llm_bench.providers.base import Usage
from engineering_llm_bench.storage.artifacts import ArtifactRef, ArtifactStore
from engineering_llm_bench.storage.db import Store, dumps, utcnow
from engineering_llm_bench.tools.runner import CommandRefusedError, CommandRunner
from engineering_llm_bench.workspace.manager import (
    PathEscapeError,
    Workspace,
    is_editable,
    prepare_workspace,
    resolve_inside,
    snapshot_hash,
)

MAX_IMPORT_BYTES = 50_000_000
_HUNK = re.compile(r"^@@ -\d+(,(\d+))? \+\d+(,(\d+))? @@")
_FORBIDDEN_PATCH = re.compile(r"^(?:new file mode|old mode|new mode|deleted file mode) 12", re.M)


@dataclass
class ImportResult:
    run_id: str
    attempts: int
    evaluated: int
    warnings: list[str]


def load_import_file(path: Path) -> ImportFile:
    if not path.is_file():
        raise ConfigError(f"import file not found: {path}")
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ConfigError(f"import file too large (> {MAX_IMPORT_BYTES} bytes)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    return parse_model(ImportFile, data, str(path))


def patch_paths(patch: str) -> list[str]:
    """File paths named by a unified diff's headers (hunk bodies are skipped, so removed lines
    such as ``-- comment`` cannot masquerade as ``--- path`` headers)."""
    paths: set[str] = set()
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        hunk = _HUNK.match(line)
        if hunk:
            old = int(hunk.group(2) or 1)
            new = int(hunk.group(4) or 1)
            i += 1
            while i < len(lines) and (old > 0 or new > 0):
                body = lines[i]
                if body.startswith("-"):
                    old -= 1
                elif body.startswith("+"):
                    new -= 1
                elif not body.startswith("\\"):
                    old -= 1
                    new -= 1
                i += 1
            continue
        if line.startswith(("--- ", "+++ ")):
            target = line[4:].split("\t")[0].strip()
            if target != "/dev/null":
                paths.add(target[2:] if target.startswith(("a/", "b/")) else target)
        elif line.startswith("diff --git "):
            parts = line.split()
            paths.update(p[2:] for p in parts[2:4] if p.startswith(("a/", "b/")))
        i += 1
    return sorted(paths)


def validate_patch(patch: str, task: Task, root: Path) -> list[str]:
    """Return the files a patch touches; refuse traversal, symlinks and non-editable paths."""
    if _FORBIDDEN_PATCH.search(patch):
        raise ConfigError("patch creates symlinks or changes file modes; refused")
    paths = patch_paths(patch)
    if not paths:
        raise ConfigError("patch contains no file headers (expected unified diff)")
    for p in paths:
        try:
            resolve_inside(root, p)
        except PathEscapeError as exc:
            raise ConfigError(f"unsafe path in patch: {exc}") from exc
        if not is_editable(p, task.spec.editable_paths):
            raise ConfigError(f"patch modifies non-editable path {p!r} for task {task.id}")
    return paths


def validate_overlay(overlay: dict[str, bytes | None], task: Task, root: Path) -> None:
    """Refuse traversal and writes outside the task's editable paths."""
    for rel in overlay:
        try:
            resolve_inside(root, rel)
        except PathEscapeError as exc:
            raise ConfigError(f"unsafe path in submission: {exc}") from exc
        if not is_editable(rel, task.spec.editable_paths):
            raise ConfigError(f"submission modifies non-editable path {rel!r} for task {task.id}")


def apply_overlay(overlay: dict[str, bytes | None], workspace: Workspace) -> tuple[bool, str]:
    """Write (or delete, for ``None``) the candidate's files into a fresh fixture copy."""
    lines = []
    for rel, data in sorted(overlay.items()):
        target = resolve_inside(workspace.root, rel)
        if data is None:
            target.unlink(missing_ok=True)
            lines.append(f"deleted {rel}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            lines.append(f"wrote {rel} ({len(data)} bytes)")
    return True, "\n".join(lines) or "(no changes)"


def apply_patch(patch: str, workspace: Workspace, runner: CommandRunner) -> tuple[bool, str]:
    """Apply with ``git apply`` (no shell; paths pre-validated). Returns (ok, log)."""
    try:
        res = runner.run_trusted(
            ["git", "apply", "--whitespace=nowarn", "--recount", "-"],
            cwd=workspace.root,
            scratch=workspace.scratch,
            timeout_s=60,
            stdin_text=patch if patch.endswith("\n") else patch + "\n",
        )
    except CommandRefusedError as exc:
        return False, str(exc)
    return res.ok, res.summary()


class RunImporter:
    def __init__(
        self,
        bench: BenchmarkConfig,
        registry: TaskRegistry,
        store: Store,
        artifacts: ArtifactStore,
    ) -> None:
        self.bench = bench
        self.registry = registry
        self.store = store
        self.artifacts = artifacts

    def import_file(
        self, path: Path, evaluate_patches: bool = True, keep_transcripts: bool = False
    ) -> ImportResult:
        doc = load_import_file(path)
        return self.import_document(
            doc,
            path.parent,
            path.name,
            evaluate_patches=evaluate_patches,
            keep_transcripts=keep_transcripts,
        )

    def import_document(
        self,
        doc: ImportFile,
        base_dir: Path,
        source_name: str,
        *,
        evaluate_patches: bool = True,
        keep_transcripts: bool = False,
        overlays: dict[str, dict[str, bytes | None]] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> ImportResult:
        """Import a validated document. ``overlays`` (task id -> {path: content or None for
        deletion}) applies the candidate's files directly instead of ``git apply``-ing the
        patch; the patch is then kept as evidence only."""
        warnings: list[str] = []
        unknown = sorted({a.task_id for a in doc.attempts} - set(self.registry.tasks))
        if unknown:
            raise ConfigError(f"import references unknown task ids: {unknown}")
        meta = doc.run
        env = collect_environment()
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-imp-" + uuid.uuid4().hex[:6]
        self.store.insert_run(
            {
                "id": run_id,
                "created_at": utcnow(),
                "status": RunStatus.RUNNING.value,
                "source": "imported",
                "profile": None,
                "suite_name": self.bench.suite_name,
                "suite_version": self.bench.suite_version,
                "model_alias": meta.model_label,
                "provider": meta.provider_label,
                "provider_model_id": meta.provider_model_id,
                "endpoint_type": None,
                "remote": None if meta.remote is None else int(meta.remote),
                "harness": meta.harness,
                "harness_version": meta.harness_version,
                "mode": ExecutionMode.MANUAL.value,
                "model_config_json": None,
                "harness_config_json": dumps(meta.model_dump(mode="json")),
                "run_config_json": dumps(
                    {
                        "import_file": source_name,
                        "evaluated": evaluate_patches,
                        **(run_config or {}),
                    }
                ),
                "config_fingerprint": fingerprint(
                    meta.model_dump(mode="json", exclude={"notes", "captured_at"})
                ),
                "env_json": dumps(env),
                "env_fingerprint": environment_fingerprint(env),
                "task_count": len({a.task_id for a in doc.attempts}),
                "notes": meta.notes,
            }
        )
        runner = CommandRunner(self.bench.limits, self.bench.isolation.network)
        evaluated = 0
        for item in doc.attempts:
            task = self.registry.get(item.task_id)
            if item.task_version and item.task_version != task.spec.version:
                warnings.append(
                    f"{item.task_id}: imported against task v{item.task_version}, catalogue has "
                    f"v{task.spec.version} - results are not directly comparable"
                )
            patch = item.patch
            if item.patch_file is not None:
                try:
                    patch_path = resolve_inside(base_dir, item.patch_file)
                except PathEscapeError as exc:
                    raise ConfigError(f"patch_file: {exc}") from exc
                patch = patch_path.read_text(encoding="utf-8")
            overlay = (overlays or {}).get(item.task_id)
            evaluated += self._import_attempt(
                run_id, task, item, patch, runner, evaluate_patches, keep_transcripts, overlay
            )
        self.store.update_run(run_id, status=RunStatus.COMPLETED.value, finished_at=utcnow())
        return ImportResult(run_id, len(doc.attempts), evaluated, warnings)

    def _import_attempt(
        self,
        run_id: str,
        task: Task,
        item: ImportedAttempt,
        patch: str | None,
        runner: CommandRunner,
        evaluate_patches: bool,
        keep_transcripts: bool,
        overlay: dict[str, bytes | None] | None = None,
    ) -> int:
        aid = f"{run_id}--{re.sub(r'[^a-zA-Z0-9_.-]+', '-', task.id)}--r{item.repeat_index}"
        refs: list[ArtifactRef] = []
        if patch:
            refs.append(self.artifacts.write_text(run_id, aid, "imported.patch", patch))
        if item.transcript and keep_transcripts:
            refs.append(self.artifacts.write_text(run_id, aid, "transcript.txt", item.transcript))
        if item.command_logs:
            logs = json.dumps([c.model_dump() for c in item.command_logs], indent=2)
            refs.append(self.artifacts.write_text(run_id, aid, "imported_commands.json", logs))
        notes: list[str] = []
        if item.self_reported_passed is not None:
            notes.append(f"self-reported passed={item.self_reported_passed} (not trusted)")
        if item.human_notes:
            notes.append(f"human notes: {item.human_notes}")
        usage_known = item.input_tokens is not None or item.output_tokens is not None
        row: dict[str, Any] = {
            "id": aid,
            "run_id": run_id,
            "task_id": task.id,
            "task_version": item.task_version or task.spec.version,
            "task_hash": task.content_hash
            if item.task_version in (None, task.spec.version)
            else None,
            "task_domain": task.spec.domain.value,
            "evaluation_kind": task.spec.evaluation.value,
            "repeat_index": item.repeat_index,
            "started_at": None,
            "finished_at": utcnow(),
            "agent_wall_s": item.elapsed_s,
            "turns": None,
            "tool_calls": None,
            "tool_calls_ok": None,
            "malformed_outputs": None,
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "usage_reliable": int(usage_known),
            "cost_usd": item.cost_usd,
            "summary": item.summary,
        }
        if not evaluate_patches or (not patch and overlay is None):
            reason = "imported without patch" if not patch else "imported without evaluation"
            row.update(
                status=AttemptStatus.SKIPPED.value,
                skip_reason=reason,
                failure_class=FailureClass.UNAVAILABLE.value,
                notes_json=dumps(notes),
            )
            self.store.save_evaluation(row, [], [], artifact_rows(refs))
            return 0
        workspace = prepare_workspace(task)
        try:
            try:
                if overlay is not None:
                    validate_overlay(overlay, task, workspace.root)
                else:
                    validate_patch(patch or "", task, workspace.root)
            except ConfigError as exc:
                row.update(
                    status=AttemptStatus.COMPLETED.value,
                    failure_class=FailureClass.MALFORMED_OUTPUT.value,
                    error=str(exc),
                    score=None,
                    passed=0,
                    notes_json=dumps(notes),
                )
                self.store.save_evaluation(row, [], [], artifact_rows(refs))
                return 0
            if overlay is not None:
                ok, log = apply_overlay(overlay, workspace)
            else:
                ok, log = apply_patch(patch or "", workspace, runner)
            refs.append(self.artifacts.write_text(run_id, aid, "apply.log", log))
            if not ok:
                row.update(
                    status=AttemptStatus.COMPLETED.value,
                    failure_class=FailureClass.MALFORMED_OUTPUT.value,
                    error="patch does not apply to the task fixture",
                    passed=0,
                    notes_json=dumps(notes),
                )
                self.store.save_evaluation(row, [], [], artifact_rows(refs))
                return 0
            refs.append(self.artifacts.write_text(run_id, aid, "diff.patch", workspace.diff()))
            harness = HarnessResult(
                finished=True,
                summary=item.summary or "",
                usage=Usage(item.input_tokens, item.output_tokens),
            )
            t0 = time.monotonic()
            outcome = evaluate(
                task,
                workspace,
                harness,
                ExecutionMode.MANUAL,
                runner,
                self.bench.limits,
                self.bench.command_allowlist,
                self.bench.weights,
                CancelToken(),
            )
            card = outcome.scorecard
            dims, checks = evaluation_rows(outcome, self.artifacts, run_id, aid, refs)
            failure = FailureClass.NONE if card.passed else FailureClass.TESTS_FAILED
            row.update(
                status=AttemptStatus.COMPLETED.value,
                failure_class=failure.value,
                eval_wall_s=round(time.monotonic() - t0, 3),
                score=None if card.score is None else round(card.score, 3),
                coverage=round(card.coverage, 3),
                passed=None if card.passed is None else int(card.passed),
                regression_failed=None
                if card.regression_failed is None
                else int(card.regression_failed),
                baseline_hash=workspace.baseline_hash,
                final_hash=snapshot_hash(workspace.current()),
                artifact_dir=self.artifacts.attempt_dir(run_id, aid)
                .relative_to(self.artifacts.root)
                .as_posix(),
                notes_json=dumps(notes + card.notes),
            )
            self.store.save_evaluation(row, dims, checks, artifact_rows(refs))
            return 1
        finally:
            workspace.cleanup()


def export_import_format(store: Store, artifacts: ArtifactStore, run_id: str) -> dict[str, Any]:
    """Re-create the import document for a stored run (round-trip support).

    Fields that were never captured are emitted as ``null`` - never filled in.
    """
    run = store.get_run(run_id)
    if run is None:
        raise ConfigError(f"unknown run id {run_id}")
    attempts = []
    for a in store.attempts_for_run(run_id):
        names = {r["name"]: r["path"] for r in store.artifacts(a["id"])}
        patch_rel = names.get("imported.patch") or names.get("diff.patch")
        patch = artifacts.resolve(patch_rel).read_text(encoding="utf-8") if patch_rel else None
        notes = json.loads(a["notes_json"] or "[]")
        human = next((n.split(": ", 1)[1] for n in notes if n.startswith("human notes: ")), None)
        attempts.append(
            {
                "task_id": a["task_id"],
                "task_version": a["task_version"],
                "repeat_index": a["repeat_index"],
                "patch": patch or None,
                "elapsed_s": a["agent_wall_s"],
                "input_tokens": a["input_tokens"],
                "output_tokens": a["output_tokens"],
                "cost_usd": a["cost_usd"],
                "summary": a["summary"],
                "human_notes": human,
            }
        )
    meta = json.loads(run["harness_config_json"] or "{}") if run["source"] == "imported" else {}
    return {
        "schema_version": 1,
        "format": "engineering-llm-bench/import",
        "run": {
            "model_label": run["model_alias"],
            "provider_label": run["provider"],
            "provider_model_id": run["provider_model_id"],
            "harness": run["harness"],
            "harness_version": run["harness_version"],
            "mode": "manual",
            "captured_at": meta.get("captured_at"),
            "remote": None if run["remote"] is None else bool(run["remote"]),
            "notes": run["notes"],
        },
        "attempts": attempts,
    }
