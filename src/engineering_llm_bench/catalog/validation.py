"""Behavioural task validation: evaluators must reject the untouched fixture and the optional
``broken/`` overlay, and accept the ``reference/`` overlay. This proves each task's evaluator
discriminates, independently of any model."""

from __future__ import annotations

from dataclasses import dataclass, field

from engineering_llm_bench.catalog.registry import Task, check_availability
from engineering_llm_bench.config.models import BenchmarkConfig
from engineering_llm_bench.control import CancelToken
from engineering_llm_bench.domain import CheckStatus, ExecutionMode
from engineering_llm_bench.evaluation.evaluator import EvaluationOutcome, evaluate
from engineering_llm_bench.harnesses.base import HarnessResult
from engineering_llm_bench.tools.runner import CommandRunner
from engineering_llm_bench.workspace.manager import prepare_workspace, workspaces_root

REFERENCE_SUMMARY = (
    "Reference solution. Verified by running the tests. Limitations: none known for the task scope."
)


@dataclass
class TaskValidationReport:
    ok: bool = True
    skipped: str | None = None
    lines: list[str] = field(default_factory=list)


def _evaluate_variant(task: Task, bench: BenchmarkConfig, overlay: str | None) -> EvaluationOutcome:
    workspace = prepare_workspace(task)
    try:
        if overlay is not None:
            workspace.apply_overlay(task.root / overlay)
        summary = REFERENCE_SUMMARY if overlay == "reference" else ""
        runner = CommandRunner(
            bench.limits,
            bench.isolation.network,
            workspaces_root=workspaces_root(),
            filesystem=bench.isolation.filesystem,
        )
        return evaluate(
            task,
            workspace,
            HarnessResult(finished=True, summary=summary),
            ExecutionMode.DIRECT,
            runner,
            bench.limits,
            bench.command_allowlist,
            bench.weights,
            CancelToken(),
        )
    finally:
        workspace.cleanup()


def _describe(name: str, outcome: EvaluationOutcome) -> list[str]:
    score = outcome.scorecard.score
    lines = [
        f"[{name}] passed={outcome.scorecard.passed} score="
        f"{'-' if score is None else f'{score:.1f}'}"
    ]
    for c in outcome.checks:
        s = "-" if c.score is None else f"{c.score:.2f}"
        lines.append(f"    {c.check_id:<28} {c.status.value:<8} {s:>5}  {c.message[:90]}")
    return lines


def validate_task_behaviour(task: Task, bench: BenchmarkConfig) -> TaskValidationReport:
    report = TaskValidationReport()
    avail = check_availability(task)
    if not avail.available:
        report.skipped = avail.reason
        return report
    fixture = _evaluate_variant(task, bench, None)
    report.lines += _describe("fixture (must fail)", fixture)
    if fixture.scorecard.passed is not False:
        report.ok = False
        report.lines.append("  !! the untouched fixture must not pass the functional checks")
    if task.reference_dir.is_dir():
        ref = _evaluate_variant(task, bench, "reference")
        report.lines += _describe("reference (must pass)", ref)
        if ref.scorecard.passed is not True:
            report.ok = False
            report.lines.append("  !! the reference solution must pass all functional checks")
        errors = [c.check_id for c in ref.checks if c.status == CheckStatus.ERROR]
        if errors:
            report.ok = False
            report.lines.append(f"  !! evaluator errors on reference: {errors}")
    else:
        report.lines.append("  (no reference/ overlay: cannot prove the task is solvable)")
    if task.broken_dir.is_dir():
        broken = _evaluate_variant(task, bench, "broken")
        report.lines += _describe("broken (must fail)", broken)
        if broken.scorecard.passed is not False:
            report.ok = False
            report.lines.append("  !! the deliberately broken patch must fail")
    return report
