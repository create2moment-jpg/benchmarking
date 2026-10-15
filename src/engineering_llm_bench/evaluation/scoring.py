"""Transparent, weighted rubric scoring.

* A dimension's score is the weight-averaged score of its *scored* checks (``error``/``skipped``
  checks are excluded and reported). Dimensions with no scored checks are ``unassessed``.
* The aggregate (0-100) renormalises over assessed dimensions only, and the fraction of the
  total weight that was assessed is reported next to it (``coverage``), so a score built from
  fewer dimensions is visibly different from a fully assessed one.
* Code quality is not credited when the candidate changed no files.
* Workflow is scored from the harness trace (finished cleanly, tool-call success, malformed
  replies, expected artifacts) - not from model self-reports.
* Human and judge scores are stored separately (see ``reporting``) and never mixed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from engineering_llm_bench.catalog.registry import Task
from engineering_llm_bench.domain import (
    DEFAULT_WEIGHTS,
    CheckStatus,
    Dimension,
    ExecutionMode,
    ScoreSource,
)
from engineering_llm_bench.evaluation.checks import CheckResult
from engineering_llm_bench.harnesses.base import HarnessResult


@dataclass
class DimensionScore:
    dimension: Dimension
    score: float | None  # 0..1, None = unassessed
    weight: float
    source: ScoreSource
    n_checks: int
    details: str = ""

    @property
    def assessed(self) -> bool:
        return self.score is not None


@dataclass
class ScoreCard:
    dimensions: list[DimensionScore]
    score: float | None  # 0..100 over assessed dimensions
    coverage: float  # assessed weight / total weight
    passed: bool | None
    regression_failed: bool | None
    notes: list[str] = field(default_factory=list)

    def by_dimension(self) -> dict[str, float | None]:
        return {d.dimension.value: d.score for d in self.dimensions}


def workflow_score(
    harness: HarnessResult, mode: ExecutionMode, task: Task, eval_dir: Path | None
) -> tuple[float | None, str]:
    """Score tool workflow & task completion from the harness's own trace."""
    if mode == ExecutionMode.MANUAL:
        return None, "manual import: workflow not observable"
    parts: list[float] = [1.0 if harness.finished else 0.0]
    if harness.tool_calls:
        parts.append(harness.tool_calls_ok / harness.tool_calls)
    if task.spec.expected_artifacts and eval_dir is not None:
        present = sum((eval_dir / a).exists() for a in task.spec.expected_artifacts)
        parts.append(present / len(task.spec.expected_artifacts))
    score = sum(parts) / len(parts)
    penalty = min(0.5, 0.1 * harness.malformed_outputs)
    score *= 1.0 - penalty
    detail = (
        f"finished={harness.finished}, tool_ok={harness.tool_calls_ok}/{harness.tool_calls}, "
        f"malformed={harness.malformed_outputs}"
    )
    return score, detail


def score_attempt(
    task: Task,
    checks: list[CheckResult],
    weights: dict[Dimension, float] | None,
    workflow: tuple[float | None, str],
    changed: bool = True,
) -> ScoreCard:
    weights = dict(task.spec.weights or weights or DEFAULT_WEIGHTS)
    dims: list[DimensionScore] = []
    notes: list[str] = []
    for dim in Dimension:
        w = weights.get(dim, 0.0)
        if dim == Dimension.WORKFLOW:
            wscore, detail = workflow
            dims.append(DimensionScore(dim, wscore, w, ScoreSource.DETERMINISTIC, 0, detail))
            continue
        relevant = [c for c in checks if c.dimension == dim]
        scored = [c for c in relevant if c.scored]
        for c in relevant:
            if c.status == CheckStatus.ERROR:
                notes.append(f"evaluator error in {c.check_id}: {c.message}")
        if not scored:
            dims.append(DimensionScore(dim, None, w, ScoreSource.DETERMINISTIC, 0, "unassessed"))
            continue
        total_w = sum(c.weight for c in scored)
        value = sum(c.weight * (c.score or 0.0) for c in scored) / total_w
        if dim == Dimension.QUALITY and not changed:
            # Linting untouched fixture code would reward doing nothing.
            dims.append(
                DimensionScore(
                    dim,
                    0.0,
                    w,
                    ScoreSource.DETERMINISTIC,
                    len(scored),
                    "no files changed: quality not credited",
                )
            )
            continue
        sources = {c.source for c in scored}
        source = (
            ScoreSource.HEURISTIC if ScoreSource.HEURISTIC in sources else ScoreSource.DETERMINISTIC
        )
        detail = "; ".join(f"{c.check_id}={c.score:.2f}" for c in scored)
        dims.append(DimensionScore(dim, value, w, source, len(scored), detail))

    total_weight = sum(d.weight for d in dims)
    assessed = [d for d in dims if d.assessed and d.weight > 0]
    assessed_weight = sum(d.weight for d in assessed)
    score = (
        100.0 * sum(d.weight * (d.score or 0.0) for d in assessed) / assessed_weight
        if assessed_weight > 0
        else None
    )
    functional = [c for c in checks if c.dimension == Dimension.FUNCTIONAL]
    scored_functional = [c for c in functional if c.scored]
    passed: bool | None = (
        all(c.status == CheckStatus.PASS for c in scored_functional) if scored_functional else None
    )
    guards = [c for c in checks if c.regression_guard and c.scored]
    regression_failed = any(c.status != CheckStatus.PASS for c in guards) if guards else None
    return ScoreCard(
        dimensions=dims,
        score=score,
        coverage=assessed_weight / total_weight if total_weight else 0.0,
        passed=passed,
        regression_failed=regression_failed,
        notes=notes,
    )
