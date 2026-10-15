"""Scoring rules and evaluator checks that do not need a full run."""

from __future__ import annotations

import sqlite3

import pytest

from engineering_llm_bench.catalog.registry import TaskRegistry
from engineering_llm_bench.domain import CheckStatus, Dimension, ExecutionMode, ScoreSource
from engineering_llm_bench.evaluation.checks import CheckResult, run_duckdb, run_sqlite
from engineering_llm_bench.evaluation.scoring import score_attempt, workflow_score
from engineering_llm_bench.harnesses.base import HarnessResult


def check(
    dim: Dimension,
    status: CheckStatus,
    score: float | None,
    weight: float = 1.0,
    guard: bool = False,
    source: ScoreSource = ScoreSource.DETERMINISTIC,
) -> CheckResult:
    return CheckResult(
        f"c-{dim}-{status}-{score}", "pytest", dim, weight, status, score, source, guard
    )


def test_aggregate_renormalises_over_assessed(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [check(Dimension.FUNCTIONAL, CheckStatus.PASS, 1.0)]
    card = score_attempt(task, checks, None, (None, "n/a"))
    assert card.score == 100.0
    assert card.coverage == pytest.approx(45 / 100)
    assert card.passed is True
    assert card.by_dimension()["quality"] is None


def test_partial_and_weighted(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [
        check(Dimension.FUNCTIONAL, CheckStatus.PASS, 1.0, weight=3),
        check(Dimension.FUNCTIONAL, CheckStatus.PARTIAL, 0.5, weight=1),
        check(Dimension.QUALITY, CheckStatus.FAIL, 0.0),
    ]
    card = score_attempt(task, checks, None, (1.0, "ok"))
    dims = card.by_dimension()
    assert dims["functional"] == pytest.approx(0.875)
    assert dims["quality"] == 0.0
    assert card.passed is False
    expected = 100 * (45 * 0.875 + 15 * 0.0 + 10 * 1.0) / (45 + 15 + 10)
    assert card.score == pytest.approx(expected)


def test_evaluator_errors_excluded_not_counted_as_failure(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [
        check(Dimension.FUNCTIONAL, CheckStatus.PASS, 1.0),
        check(Dimension.QUALITY, CheckStatus.ERROR, None),
        check(Dimension.ROBUSTNESS, CheckStatus.SKIPPED, None),
    ]
    card = score_attempt(task, checks, None, (None, ""))
    assert card.score == 100.0
    assert card.by_dimension()["quality"] is None
    assert any("evaluator error" in n for n in card.notes)


def test_no_functional_scored_means_unknown_pass(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    card = score_attempt(
        task, [check(Dimension.FUNCTIONAL, CheckStatus.ERROR, None)], None, (None, "")
    )
    assert card.passed is None and card.score is None


def test_no_change_gets_no_quality_credit(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [
        check(Dimension.QUALITY, CheckStatus.PASS, 1.0),
        check(Dimension.FUNCTIONAL, CheckStatus.FAIL, 0.0),
    ]
    assert (
        score_attempt(task, checks, None, (None, ""), changed=False).by_dimension()["quality"]
        == 0.0
    )
    assert (
        score_attempt(task, checks, None, (None, ""), changed=True).by_dimension()["quality"] == 1.0
    )


def test_regression_guard(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [check(Dimension.FUNCTIONAL, CheckStatus.FAIL, 0.5, guard=True)]
    assert score_attempt(task, checks, None, (None, "")).regression_failed is True


def test_heuristic_source_propagates(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    checks = [
        check(Dimension.FUNCTIONAL, CheckStatus.PASS, 1.0),
        check(Dimension.EXPLANATION, CheckStatus.PASS, 1.0, source=ScoreSource.HEURISTIC),
    ]
    card = score_attempt(task, checks, None, (None, ""))
    expl = next(d for d in card.dimensions if d.dimension == Dimension.EXPLANATION)
    assert expl.source == ScoreSource.HEURISTIC


def test_workflow_score(registry: TaskRegistry) -> None:
    task = registry.get("python.fix-interval-merge")
    perfect = HarnessResult(finished=True, tool_calls=4, tool_calls_ok=4)
    assert workflow_score(perfect, ExecutionMode.AGENT, task, None)[0] == 1.0
    sloppy = HarnessResult(finished=False, tool_calls=4, tool_calls_ok=2, malformed_outputs=2)
    assert workflow_score(sloppy, ExecutionMode.AGENT, task, None)[0] == pytest.approx(0.25 * 0.8)
    assert workflow_score(perfect, ExecutionMode.MANUAL, task, None)[0] is None


SETUP = "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2);"


def test_sqlite_runs_select() -> None:
    cols, rows = run_sqlite(SETUP, "SELECT a FROM t ORDER BY a", 5)
    assert cols == ["a"] and rows == [(1,), (2,)]


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE t",
        "INSERT INTO t VALUES (3)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
        "PRAGMA writable_schema=1",
    ],
)
def test_sqlite_candidate_cannot_modify_or_attach(query: str) -> None:
    with pytest.raises(sqlite3.DatabaseError):
        run_sqlite(SETUP, query, 5)


def test_sqlite_runaway_query_interrupted() -> None:
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(sqlite3.OperationalError):
        run_sqlite(SETUP, slow, 0.5)


def test_duckdb_external_access_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("duckdb")
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    with pytest.raises(Exception, match=r"(?i)external|permission|disabled"):
        run_duckdb("CREATE TABLE t (a INT);", f"SELECT * FROM read_csv_auto('{secret}')", 5)
