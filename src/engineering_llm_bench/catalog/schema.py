"""Versioned task definition schema (``task.yaml``).

A task directory contains::

    task.yaml     metadata, prompt, acceptance criteria, evaluator checks (this schema)
    fixture/      initial workspace given to the model (copied fresh for every attempt)
    evaluator/    hidden evaluator inputs (tests, seed data, expected results) - never shown
    reference/    reference solution overlay, used only by ``bench task validate`` and the
                  clearly labelled fake provider
    broken/       optional deliberately broken overlay used to prove the evaluator rejects it

Task files are treated as untrusted input: paths must be relative and stay inside the task or
workspace, and evaluator commands may only start allowlisted executables.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engineering_llm_bench.domain import (
    Difficulty,
    Dimension,
    Domain,
    EvaluationKind,
    Profile,
    ScoreSource,
    validate_weights,
)

TASK_SCHEMA_VERSION = 1
_TASK_ID_RE = r"^[a-z0-9_]+\.[a-z0-9][a-z0-9_-]{1,63}$"
_CHECK_ID_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"

#: Placeholders that evaluator argv may use. ``{python}`` is the benchmark interpreter.
ARGV_PLACEHOLDERS = ("{python}", "{hidden}", "{workspace}")


def safe_relpath(value: str) -> str:
    """Validate a POSIX relative path that cannot escape its root."""
    if not value or "\x00" in value or "\\" in value:
        raise ValueError(f"invalid path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative without '..': {value!r}")
    return value


def _safe_relpaths(value: list[str]) -> list[str]:
    return [safe_relpath(v) for v in value]


ToolName = Literal["list_files", "read_file", "write_file", "run_command"]


def _all_tools() -> list[ToolName]:
    return ["list_files", "read_file", "write_file", "run_command"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CheckBase(_Strict):
    id: Annotated[str, Field(pattern=_CHECK_ID_RE)]
    dimension: Dimension
    weight: float = Field(default=1.0, gt=0, le=100)
    description: str = ""
    #: A check guarding existing behaviour; failures count towards the regression rate.
    regression_guard: bool = False
    source: ScoreSource = ScoreSource.DETERMINISTIC
    timeout_s: float | None = Field(default=None, gt=0, le=3600)


class PytestCheck(_CheckBase):
    """Run pytest on ``targets`` (relative to the evaluation root; hidden files live in
    ``_hidden/``). Score = passed / collected tests when ``partial_credit``."""

    kind: Literal["pytest"]
    targets: list[str] = Field(min_length=1)
    partial_credit: bool = True

    _targets = field_validator("targets")(_safe_relpaths)


class CommandCheck(_CheckBase):
    """Run an allowlisted command. ``parse=check_lines`` scores lines ``CHECK <name> PASS|FAIL``."""

    kind: Literal["command"]
    argv: list[str] = Field(min_length=1)
    expect_exit: int = 0
    parse: Literal["exit", "check_lines"] = "exit"


class CppTestCheck(_CheckBase):
    """Compile C++ sources (candidate code + hidden test) and run the binary. The test prints
    ``CHECK <name> PASS|FAIL`` lines; score = passed / max(reported, expected_checks)."""

    kind: Literal["cpp_test"]
    sources: list[str] = Field(min_length=1)
    include_dirs: list[str] = Field(default_factory=lambda: ["."])
    std: Literal["c++17", "c++20"] = "c++17"
    flags: list[str] = Field(default_factory=lambda: ["-O1"])
    compiler: Literal["g++", "c++", "clang++"] = "g++"
    expected_checks: int | None = Field(default=None, ge=1)

    _sources = field_validator("sources", "include_dirs")(_safe_relpaths)

    @field_validator("flags")
    @classmethod
    def _flags(cls, value: list[str]) -> list[str]:
        for flag in value:
            if not flag.startswith(("-O", "-W", "-D", "-g", "-f", "-pthread")) or ".." in flag:
                raise ValueError(f"unsupported compiler flag {flag!r}")
        return value


class RuffCheck(_CheckBase):
    """Static lint of the candidate's Python files. Score = 1 / (1 + 0.25 * violations)."""

    kind: Literal["ruff"]
    targets: list[str] = Field(min_length=1)
    select: list[str] = Field(default_factory=lambda: ["E", "F", "W", "B", "UP", "SIM"])
    line_length: int = Field(default=100, ge=60, le=200)

    _targets = field_validator("targets")(_safe_relpaths)


class SqlResultCheck(_CheckBase):
    """Execute the candidate query against a freshly seeded DB and compare to expected rows."""

    kind: Literal["sql_result"]
    engine: Literal["sqlite", "duckdb"] = "sqlite"
    query_file: str
    setup_sql: str = Field(description="Hidden SQL script that creates and seeds the database.")
    expected_csv: str = Field(description="Hidden CSV with header row of expected results.")
    ordered: bool = False
    float_tolerance: float = Field(default=1e-6, ge=0)

    _paths = field_validator("query_file", "setup_sql", "expected_csv")(safe_relpath)


class TestsDetectBugCheck(_CheckBase):
    """The candidate's tests must FAIL on the original (buggy) fixture and PASS on the final
    workspace - i.e. they are genuine regression tests for the defect."""

    kind: Literal["tests_detect_bug"]
    test_targets: list[str] = Field(min_length=1)

    _targets = field_validator("test_targets")(_safe_relpaths)


class AddedTestsCheck(_CheckBase):
    """Count test definitions matching ``pattern`` under ``glob``; pass if the count increased."""

    kind: Literal["added_tests"]
    glob: str = "tests/**/*.py"
    pattern: str = r"^\s*def test_"
    min_added: int = Field(default=1, ge=1)


class RubricCriterion(_Strict):
    id: Annotated[str, Field(pattern=_CHECK_ID_RE)]
    description: str
    any_of: list[str] = Field(default_factory=list, description="Regexes; one must match.")
    all_of: list[str] = Field(default_factory=list, description="Regexes; all must match.")
    none_of: list[str] = Field(default_factory=list, description="Regexes; none may match.")
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _non_empty(self) -> RubricCriterion:
        if not (self.any_of or self.all_of or self.none_of):
            raise ValueError(f"criterion {self.id!r} needs any_of/all_of/none_of patterns")
        return self


class TextRubricCheck(_CheckBase):
    """Deterministic text rubric over a written artifact (e.g. review.md). Each criterion is an
    explicit, inspectable property; the evidence records which patterns matched."""

    kind: Literal["text_rubric"]
    file: str
    criteria: list[RubricCriterion] = Field(min_length=1)
    max_bytes: int = Field(default=200_000, ge=100)

    _file = field_validator("file")(safe_relpath)


class SummaryCheck(_CheckBase):
    """Heuristic handoff-quality check on the agent's final summary (presence, length,
    mentions of testing and limitations). Marked ``heuristic``; human review can supersede."""

    kind: Literal["summary"]
    source: ScoreSource = ScoreSource.HEURISTIC
    max_words: int = Field(default=400, ge=20)


class FilesExistCheck(_CheckBase):
    kind: Literal["files_exist"]
    paths: list[str] = Field(min_length=1)

    _paths = field_validator("paths")(_safe_relpaths)


CheckSpec = Annotated[
    PytestCheck
    | CommandCheck
    | CppTestCheck
    | RuffCheck
    | SqlResultCheck
    | TestsDetectBugCheck
    | AddedTestsCheck
    | TextRubricCheck
    | SummaryCheck
    | FilesExistCheck,
    Field(discriminator="kind"),
]


class Requirements(_Strict):
    executables: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)


class TaskSpec(_Strict):
    """A versioned benchmark task."""

    schema_version: Literal[1] = 1
    id: Annotated[str, Field(pattern=_TASK_ID_RE)]
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    title: str = Field(min_length=3, max_length=200)
    domain: Domain
    difficulty: Difficulty
    evaluation: EvaluationKind = EvaluationKind.EXECUTABLE
    estimated_runtime_s: int = Field(ge=1, le=7200)
    timeout_s: int = Field(ge=10, le=4 * 3600)
    resource_profile: Literal["small", "medium", "large"] = "small"
    skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    profiles: list[Profile] = Field(min_length=1)
    requires: Requirements = Requirements()
    allowed_tools: list[ToolName] = Field(default_factory=_all_tools)
    allowed_commands: list[str] = Field(default_factory=lambda: ["python", "pytest"])
    editable_paths: list[str] = Field(default_factory=lambda: ["**"])
    expected_artifacts: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=20)
    acceptance_criteria: list[str] = Field(min_length=1)
    weights: dict[Dimension, float] | None = None
    human_review_dimensions: list[Dimension] = Field(default_factory=list)
    checks: list[CheckSpec] = Field(min_length=1)

    @field_validator("expected_artifacts")
    @classmethod
    def _artifacts(cls, value: list[str]) -> list[str]:
        return [safe_relpath(v) for v in value]

    @field_validator("weights")
    @classmethod
    def _weights(cls, value: dict[Dimension, float] | None) -> dict[Dimension, float] | None:
        if value is not None:
            validate_weights(value)
        return value

    @model_validator(mode="after")
    def _consistency(self) -> TaskSpec:
        ids = [c.id for c in self.checks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate check ids: {dupes}")
        if not any(c.dimension == Dimension.FUNCTIONAL for c in self.checks):
            raise ValueError("a task needs at least one functional check")
        if Profile.SMOKE in self.profiles and Profile.STANDARD not in self.profiles:
            raise ValueError("smoke tasks must also be in the standard profile")
        return self

    @property
    def domain_label(self) -> str:
        return self.domain.value
