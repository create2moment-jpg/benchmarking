"""Independent evaluation of a finished attempt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engineering_llm_bench.catalog.registry import Task
from engineering_llm_bench.config.models import LimitsConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import Dimension, ExecutionMode
from engineering_llm_bench.evaluation.checks import CheckResult, EvalContext, run_check
from engineering_llm_bench.evaluation.scoring import ScoreCard, score_attempt, workflow_score
from engineering_llm_bench.harnesses.base import HarnessResult
from engineering_llm_bench.tools.runner import CommandRunner
from engineering_llm_bench.workspace.manager import Workspace

_MEMORY = {"small": 1, "medium": 2}


@dataclass
class EvaluationOutcome:
    checks: list[CheckResult]
    scorecard: ScoreCard
    eval_dir: Path
    cpu_s: float | None = None
    max_rss_mb: float | None = None


def evaluator_allowlist(task: Task, global_allowlist: list[str]) -> set[str]:
    """Evaluator commands may use the benchmark allowlist plus the pinned Python tools."""
    return set(global_allowlist) | {"python", "pytest", "ruff"}


def evaluate(
    task: Task,
    workspace: Workspace,
    harness: HarnessResult,
    mode: ExecutionMode,
    runner: CommandRunner,
    limits: LimitsConfig,
    allowlist: list[str],
    weights: dict[Dimension, float] | None,
    cancel: CancelToken,
) -> EvaluationOutcome:
    eval_dir = workspace.make_eval_copy()
    factor = _MEMORY.get(task.spec.resource_profile)
    ctx = EvalContext(
        task=task,
        workspace=workspace,
        eval_dir=eval_dir,
        runner=runner,
        limits=limits,
        cancel=cancel,
        allowlist=evaluator_allowlist(task, allowlist),
        summary=harness.summary,
        memory_mb=limits.memory_mb * factor if factor else None,
    )
    results = [run_check(spec, ctx) for spec in task.spec.checks]
    changed = bool(workspace.changed_files())
    card = score_attempt(
        task, results, weights, workflow_score(harness, mode, task, eval_dir), changed
    )
    return EvaluationOutcome(results, card, eval_dir, round(ctx.cpu_s, 3), ctx.max_rss_mb or None)
