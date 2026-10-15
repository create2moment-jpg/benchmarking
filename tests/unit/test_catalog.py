"""Task schema validation and registry selection."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from engineering_llm_bench.catalog.registry import TaskRegistry, hash_tree, load_task
from engineering_llm_bench.catalog.schema import TaskSpec
from engineering_llm_bench.config.loader import ConfigError, parse_model
from engineering_llm_bench.domain import Domain, Profile
from tests.conftest import REPO

BASE: dict[str, Any] = yaml.safe_load(
    (REPO / "tasks/python/fix_interval_merge/task.yaml").read_text()
)


def spec_with(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE)
    data.update(changes)
    return data


def test_catalogue_has_all_core_domains(registry: TaskRegistry) -> None:
    domains = {t.spec.domain for t in registry.tasks.values()}
    required = {
        Domain.PYTHON,
        Domain.CPP_SENSOR_FUSION,
        Domain.PERCEPTION,
        Domain.SQL,
        Domain.SPARK,
        Domain.PYTORCH,
        Domain.BUILD,
        Domain.DEBUGGING,
        Domain.CODE_REVIEW,
        Domain.PRODUCTION,
        Domain.REASONING,
    }
    assert required <= domains


def test_profile_sizes(registry: TaskRegistry) -> None:
    smoke = registry.select(Profile.SMOKE)
    standard = registry.select(Profile.STANDARD)
    extended = registry.select(Profile.EXTENDED)
    assert 3 <= len(smoke) <= 5
    assert 10 <= len(standard) <= 15
    assert len(extended) == len(registry.tasks)
    assert {t.id for t in smoke} <= {t.id for t in standard}


@pytest.mark.parametrize(
    "check",
    [
        {"id": "x", "kind": "pytest", "dimension": "functional", "targets": ["../outside.py"]},
        {"id": "x", "kind": "pytest", "dimension": "functional", "targets": ["/etc/passwd"]},
        {
            "id": "x",
            "kind": "text_rubric",
            "dimension": "functional",
            "file": "a/../../b",
            "criteria": [{"id": "c", "description": "d", "any_of": ["x"]}],
        },
    ],
)
def test_path_traversal_in_checks_rejected(check: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="relative"):
        parse_model(TaskSpec, spec_with(checks=[check]), "task")


def test_duplicate_check_ids() -> None:
    check = {"id": "a", "kind": "pytest", "dimension": "functional", "targets": ["tests"]}
    with pytest.raises(ConfigError, match="duplicate check ids"):
        parse_model(TaskSpec, spec_with(checks=[check, check]), "task")


def test_requires_functional_check() -> None:
    check = {"id": "a", "kind": "ruff", "dimension": "quality", "targets": ["x.py"]}
    with pytest.raises(ConfigError, match="functional"):
        parse_model(TaskSpec, spec_with(checks=[check]), "task")


def test_bad_task_id_and_version() -> None:
    with pytest.raises(ConfigError):
        parse_model(TaskSpec, spec_with(id="No Dots"), "task")
    with pytest.raises(ConfigError):
        parse_model(TaskSpec, spec_with(version="1.0"), "task")


def test_compiler_flags_restricted() -> None:
    check = {
        "id": "a",
        "kind": "cpp_test",
        "dimension": "functional",
        "sources": ["a.cpp"],
        "flags": ["-fplugin=/tmp/evil.so", "-o/etc/x"],
    }
    with pytest.raises(ConfigError):
        parse_model(TaskSpec, spec_with(checks=[check]), "task")


def test_hash_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1")
    first = hash_tree(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("ignored")
    assert hash_tree(tmp_path) == first
    (tmp_path / "a.txt").write_text("2")
    assert hash_tree(tmp_path) != first


def test_missing_fixture_dir(tmp_path: Path) -> None:
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(BASE))
    with pytest.raises(ConfigError, match="fixture"):
        load_task(tmp_path / "task.yaml")


def test_duplicate_task_ids(tmp_path: Path) -> None:
    for name in ("a", "b"):
        d = tmp_path / name
        (d / "fixture").mkdir(parents=True)
        (d / "task.yaml").write_text(yaml.safe_dump(BASE))
    with pytest.raises(ConfigError, match="duplicate task id"):
        TaskRegistry.discover(tmp_path)


def test_select_filters(registry: TaskRegistry) -> None:
    assert all(
        t.spec.domain == Domain.SQL for t in registry.select(Profile.STANDARD, domains=["sql"])
    )
    assert [t.id for t in registry.select(Profile.SMOKE, task_ids=["sql.sensor-gaps"])] == [
        "sql.sensor-gaps"
    ]
