"""Workspace tools exposed to the built-in agent harness (and used by the direct harness)."""

from __future__ import annotations

from dataclasses import dataclass

from engineering_llm_bench.harnesses.base import HarnessContext
from engineering_llm_bench.harnesses.protocol import AgentAction
from engineering_llm_bench.tools.runner import CommandRefusedError
from engineering_llm_bench.workspace.manager import (
    HIDDEN_DIRNAME,
    PathEscapeError,
    is_editable,
    iter_files,
    resolve_inside,
)

_MAX_LIST = 500


class ToolError(Exception):
    """A tool call was refused or failed; the message is shown to the model."""


@dataclass(frozen=True)
class ToolOutcome:
    observation: str
    ok: bool
    exit_code: int | None = None


def _rel(ctx: HarnessContext, path: str) -> str:
    try:
        target = resolve_inside(ctx.workspace.root, path)
    except PathEscapeError as exc:
        raise ToolError(str(exc)) from exc
    return target.relative_to(ctx.workspace.root.resolve()).as_posix() or "."


def write_file(ctx: HarnessContext, path: str, content: str) -> str:
    rel = _rel(ctx, path)
    if rel == "." or rel.split("/")[0] == HIDDEN_DIRNAME:
        raise ToolError(f"cannot write {path!r}")
    if not is_editable(rel, ctx.task.spec.editable_paths):
        raise ToolError(
            f"{rel!r} is not editable; allowed patterns: {ctx.task.spec.editable_paths}"
        )
    data = content.encode("utf-8")
    if len(data) > ctx.limits.max_file_bytes:
        raise ToolError(f"file too large ({len(data)} bytes > {ctx.limits.max_file_bytes})")
    target = ctx.workspace.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return rel


def execute(ctx: HarnessContext, action: AgentAction) -> ToolOutcome:
    """Execute one non-finish action. Policy violations become failed observations."""
    try:
        if action.action == "list_files":
            rel = _rel(ctx, action.path or ".")
            base = ctx.workspace.root / rel if rel != "." else ctx.workspace.root
            if not base.is_dir():
                raise ToolError(f"not a directory: {action.path}")
            files = [p.relative_to(ctx.workspace.root).as_posix() for p in iter_files(base)]
            shown = files[:_MAX_LIST]
            more = f"\n... {len(files) - _MAX_LIST} more" if len(files) > _MAX_LIST else ""
            return ToolOutcome("\n".join(shown) + more or "(empty)", True)
        if action.action == "read_file":
            rel = _rel(ctx, action.path or "")
            target = ctx.workspace.root / rel
            if not target.is_file() or target.is_symlink():
                raise ToolError(f"no such file: {action.path}")
            if target.stat().st_size > ctx.limits.max_file_bytes:
                raise ToolError("file too large to read")
            text = target.read_text(encoding="utf-8", errors="replace")
            return ToolOutcome(text, True)
        if action.action == "write_file":
            rel = write_file(ctx, action.path or "", action.content or "")
            return ToolOutcome(f"wrote {rel} ({len(action.content or '')} chars)", True)
        if action.action == "run_command":
            timeout = min(ctx.limits.command_timeout_s, max(1.0, ctx.budget.remaining_s()))
            result = ctx.runner.run(
                list(action.argv),
                cwd=ctx.workspace.root,
                root=ctx.workspace.root,
                scratch=ctx.workspace.scratch,
                allowed=ctx.allowed_commands,
                timeout_s=timeout,
                cancel=ctx.cancel,
            )
            ok = not (result.timed_out or result.cancelled or result.exit_code is None)
            return ToolOutcome(
                result.summary(ctx.limits.max_observation_chars), ok, result.exit_code
            )
    except (ToolError, CommandRefusedError) as exc:
        return ToolOutcome(f"ERROR: {exc}", False)
    except OSError as exc:
        return ToolOutcome(f"ERROR: {type(exc).__name__}: {exc}", False)
    return ToolOutcome(f"ERROR: unsupported action {action.action}", False)
