"""Acceptance 4 and 5: every task is independently validated - its evaluator rejects the
untouched fixture (and the deliberately broken patch, when provided) and accepts the reference
solution. Tasks whose optional runtime is missing are reported as skipped with the reason."""

from __future__ import annotations

import pytest

from engineering_llm_bench.catalog.registry import TaskRegistry
from engineering_llm_bench.catalog.validation import validate_task_behaviour
from engineering_llm_bench.config.loader import load_benchmark
from tests.conftest import REPO

pytestmark = pytest.mark.integration

REGISTRY = TaskRegistry.discover(REPO / "tasks")
BENCH = load_benchmark(REPO / "configs" / "benchmark.example.yaml")


@pytest.mark.parametrize("task_id", sorted(REGISTRY.tasks))
def test_task_behaviour(task_id: str) -> None:
    report = validate_task_behaviour(REGISTRY.get(task_id), BENCH)
    if report.skipped:
        pytest.skip(f"{task_id}: {report.skipped}")
    assert report.ok, "\n".join(report.lines)
