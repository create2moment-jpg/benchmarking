"""Core, provider-independent domain types shared across the benchmark.

These types are the internal data contracts. Anything crossing a trust boundary (config files,
task files, imported runs) is validated with pydantic models defined in ``config`` /
``catalog`` / ``importing``; the enums below are reused there so values stay consistent.
"""

from __future__ import annotations

from enum import StrEnum


class Dimension(StrEnum):
    """Scoring dimensions. Each task maps evaluator checks onto these."""

    FUNCTIONAL = "functional"
    REGRESSION = "regression"
    QUALITY = "quality"
    ROBUSTNESS = "robustness"
    WORKFLOW = "workflow"
    EXPLANATION = "explanation"


#: Default dimension weights for executable coding tasks (percent; normalised at scoring time).
DEFAULT_WEIGHTS: dict[Dimension, float] = {
    Dimension.FUNCTIONAL: 45.0,
    Dimension.REGRESSION: 15.0,
    Dimension.QUALITY: 15.0,
    Dimension.ROBUSTNESS: 10.0,
    Dimension.WORKFLOW: 10.0,
    Dimension.EXPLANATION: 5.0,
}


class Domain(StrEnum):
    """Skill domains covered by the task catalogue."""

    PYTHON = "python"
    CPP_SENSOR_FUSION = "cpp_sensor_fusion"
    PERCEPTION = "perception"
    SQL = "sql"
    SPARK = "spark"
    PYTORCH = "pytorch"
    BUILD = "build"
    DEBUGGING = "debugging"
    CODE_REVIEW = "code_review"
    PRODUCTION = "production"
    REASONING = "reasoning"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Profile(StrEnum):
    """Run profiles trading coverage for wall time / cost."""

    SMOKE = "smoke"
    STANDARD = "standard"
    EXTENDED = "extended"


class EvaluationKind(StrEnum):
    """How a task is evaluated; shown in the catalogue so readers know what a score means."""

    EXECUTABLE = "executable"  # deterministic evaluator commands / tests decide
    RUBRIC = "rubric"  # deterministic text-property rubric (+ optional human/judge review)
    HUMAN_REVIEW = "human_review"  # requires human review to be scored meaningfully


class ExecutionMode(StrEnum):
    """Separate execution paths; never compared silently against each other."""

    DIRECT = "direct"  # (A) single model completion returning files
    AGENT = "agent"  # (B) plan / edit / run tools loop (built-in or external CLI harness)
    MANUAL = "manual"  # (C) evidence imported from a workflow that cannot be automated


class AttemptStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"  # harness finished (successfully or not) and evaluators ran
    FAILED = "failed"  # infrastructure failure prevented a meaningful evaluation
    SKIPPED = "skipped"  # unavailable runtime / dependency; never counted as pass or fail
    INTERRUPTED = "interrupted"  # cancelled or process stopped; inspectable, resumable


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class FailureClass(StrEnum):
    """Why an attempt did not succeed. Separates model, harness and infrastructure causes."""

    NONE = "none"
    TESTS_FAILED = "tests_failed"  # model output evaluated and functional checks failed
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_ERROR = "provider_error"
    MALFORMED_OUTPUT = "malformed_output"
    HARNESS_ERROR = "harness_error"
    EVALUATION_ERROR = "evaluation_error"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


class CheckStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - not a password
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    SKIPPED = "skipped"


class ScoreSource(StrEnum):
    """Provenance of a score. Judge and human scores are never silently mixed with deterministic."""

    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    HUMAN = "human"
    JUDGE = "judge"


def validate_weights(weights: dict[Dimension, float]) -> None:
    """Raise ``ValueError`` for negative, non-finite or all-zero weights."""
    for dim, w in weights.items():
        if not (w >= 0) or w == float("inf"):
            raise ValueError(f"weight for {dim} must be a finite non-negative number, got {w}")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one weight must be positive")
