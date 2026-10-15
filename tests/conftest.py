"""Shared fixtures: a benchmark environment rooted in a temp dir using the repo's task catalogue."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from engineering_llm_bench.catalog.registry import TaskRegistry
from engineering_llm_bench.config.loader import load_benchmark, load_models
from engineering_llm_bench.config.models import BenchmarkConfig, ModelsFile
from engineering_llm_bench.orchestration.runner import BenchmarkRunner, build_redactor
from engineering_llm_bench.storage.artifacts import ArtifactStore
from engineering_llm_bench.storage.db import Store

REPO = Path(__file__).resolve().parents[1]
FAST_TASKS = ["python.fix-interval-merge", "sql.latest-sensor-reading"]


@dataclass
class BenchEnv:
    root: Path
    bench: BenchmarkConfig
    models: ModelsFile
    registry: TaskRegistry
    store: Store
    artifacts: ArtifactStore

    def runner(self) -> BenchmarkRunner:
        return BenchmarkRunner(self.bench, self.models, self.registry, self.store, self.artifacts)


@pytest.fixture(scope="session")
def registry() -> TaskRegistry:
    return TaskRegistry.discover(REPO / "tasks")


@pytest.fixture
def bench_env(tmp_path: Path, registry: TaskRegistry) -> Iterator[BenchEnv]:
    bench = load_benchmark(REPO / "configs" / "benchmark.example.yaml").model_copy(
        update={
            "tasks_dir": str(REPO / "tasks"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "database": str(tmp_path / "artifacts" / "bench.db"),
        }
    )
    models = load_models(REPO / "configs" / "models.example.yaml")
    store = Store(tmp_path / "artifacts" / "bench.db")
    artifacts = ArtifactStore(tmp_path / "artifacts", build_redactor(models, bench))
    env = BenchEnv(tmp_path, bench, models, registry, store, artifacts)
    yield env
    store.close()
