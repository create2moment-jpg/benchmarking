"""Acceptance 2: config validation catches missing fields, invalid weights, unknown task ids,
malformed endpoint URLs and unsafe limits."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from engineering_llm_bench.catalog.registry import TaskRegistry
from engineering_llm_bench.config.loader import (
    ConfigError,
    load_benchmark,
    load_models,
    parse_model,
    read_yaml,
)
from engineering_llm_bench.config.models import BenchmarkConfig, ModelsFile
from engineering_llm_bench.domain import Profile
from tests.conftest import REPO

BENCH = read_yaml(REPO / "configs" / "benchmark.example.yaml")
MODELS = read_yaml(REPO / "configs" / "models.example.yaml")


def bench_with(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(BENCH)
    data.update(changes)
    return data


def test_example_configs_are_valid() -> None:
    assert load_benchmark(REPO / "configs" / "benchmark.example.yaml").suite_version
    models = load_models(REPO / "configs" / "models.example.yaml")
    assert models.model("fake-reference").provider == "fake"


def test_missing_required_field() -> None:
    data = bench_with()
    del data["suite_version"]
    with pytest.raises(ConfigError, match="suite_version"):
        parse_model(BenchmarkConfig, data, "bench")


def test_missing_profile() -> None:
    data = bench_with()
    del data["profiles"]["extended"]
    with pytest.raises(ConfigError, match="profiles missing"):
        parse_model(BenchmarkConfig, data, "bench")


@pytest.mark.parametrize(
    "weights",
    [{"functional": -1}, {"functional": 0, "quality": 0}, {"functional": float("inf")}],
)
def test_invalid_weights(weights: dict[str, float]) -> None:
    with pytest.raises(ConfigError, match="weight"):
        parse_model(BenchmarkConfig, bench_with(weights=weights), "bench")


def test_unknown_weight_dimension() -> None:
    with pytest.raises(ConfigError):
        parse_model(BenchmarkConfig, bench_with(weights={"speed": 10}), "bench")


def test_typo_key_rejected() -> None:
    with pytest.raises(ConfigError, match="Extra inputs"):
        parse_model(BenchmarkConfig, bench_with(allow_remote_provider=True), "bench")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("limits", "command_timeout_s", 36000),
        ("limits", "memory_mb", 10**7),
        ("limits", "command_timeout_s", 0),
        ("concurrency", "max_parallel_attempts", 1000),
    ],
)
def test_unsafe_limits(section: str, key: str, value: int) -> None:
    data = bench_with()
    data[section] = {**data.get(section, {}), key: value}
    with pytest.raises(ConfigError):
        parse_model(BenchmarkConfig, data, "bench")


def test_unsafe_profile_turns() -> None:
    data = bench_with()
    data["profiles"]["smoke"]["max_turns"] = 100000
    with pytest.raises(ConfigError, match="max_turns"):
        parse_model(BenchmarkConfig, data, "bench")


def test_task_timeout_cannot_exceed_run_timeout() -> None:
    data = bench_with()
    data["profiles"]["smoke"]["max_task_timeout_s"] = 5000
    with pytest.raises(ConfigError, match="exceeds run_timeout_s"):
        parse_model(BenchmarkConfig, data, "bench")


def test_shell_not_allowed_in_allowlist() -> None:
    with pytest.raises(ConfigError, match="shells"):
        parse_model(BenchmarkConfig, bench_with(command_allowlist=["python", "bash"]), "bench")


def models_with(model: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "models": [model]}


@pytest.mark.parametrize(
    "url", ["localhost:11434", "ftp://localhost/v1", "http://", "http://user:pw@localhost/v1"]
)
def test_malformed_endpoint_url(url: str) -> None:
    model = {"alias": "m", "provider": "openai_compatible", "model_id": "x", "base_url": url}
    with pytest.raises(ConfigError, match=r"URL|credentials"):
        parse_model(ModelsFile, models_with(model), "models")


def test_public_endpoint_requires_remote_flag() -> None:
    model = {
        "alias": "m",
        "provider": "openai_compatible",
        "model_id": "x",
        "base_url": "https://api.example.com/v1",
    }
    with pytest.raises(ConfigError, match="remote: true"):
        parse_model(ModelsFile, models_with(model), "models")
    model["remote"] = True
    assert parse_model(ModelsFile, models_with(model), "models").models[0].remote


def test_private_network_endpoint_is_local() -> None:
    model = {
        "alias": "m",
        "provider": "openai_compatible",
        "model_id": "x",
        "base_url": "http://192.168.1.20:8000/v1",
    }
    assert not parse_model(ModelsFile, models_with(model), "models").models[0].remote


def test_api_key_must_be_env_name() -> None:
    model = {
        "alias": "m",
        "provider": "openai_compatible",
        "model_id": "x",
        "base_url": "http://localhost:1/v1",
        "api_key_env": "sk-live-abcdef123456",
    }
    with pytest.raises(ConfigError, match="NAME of an environment variable"):
        parse_model(ModelsFile, models_with(model), "models")


def test_literal_authorization_header_rejected() -> None:
    model = {
        "alias": "m",
        "provider": "openai_compatible",
        "model_id": "x",
        "base_url": "http://localhost:1/v1",
        "headers": {"Authorization": "Bearer abc"},
    }
    with pytest.raises(ConfigError, match="environment variable"):
        parse_model(ModelsFile, models_with(model), "models")


def test_duplicate_aliases() -> None:
    m = {"alias": "a", "provider": "fake", "model_id": "x"}
    with pytest.raises(ConfigError, match="duplicate model aliases"):
        parse_model(ModelsFile, {"schema_version": 1, "models": [m, m]}, "models")


def test_unknown_task_ids_in_profile(registry: TaskRegistry) -> None:
    data = bench_with()
    data["profiles"]["smoke"]["include_tasks"] = ["python.does-not-exist"]
    cfg = parse_model(BenchmarkConfig, data, "bench")
    with pytest.raises(ConfigError, match="unknown task ids"):
        registry.select(Profile.SMOKE, cfg.profiles[Profile.SMOKE])
    with pytest.raises(ConfigError, match="unknown task ids"):
        registry.select(Profile.SMOKE, task_ids=["nope.task"])


def test_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "b.yaml"
    bad.write_text("profiles: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_benchmark(bad)


def test_yaml_safe_load_rejects_python_tags(tmp_path: Path) -> None:
    bad = tmp_path / "b.yaml"
    bad.write_text("!!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(ConfigError):
        load_benchmark(bad)


def test_cli_harness_needs_argv() -> None:
    with pytest.raises(ConfigError, match="requires argv"):
        parse_model(
            ModelsFile,
            {"schema_version": 1, "models": [], "harnesses": [{"name": "x", "kind": "cli"}]},
            "models",
        )


def test_example_yaml_round_trips() -> None:
    cfg = load_benchmark(REPO / "configs" / "benchmark.example.yaml")
    again = parse_model(
        BenchmarkConfig, yaml.safe_load(yaml.safe_dump(cfg.model_dump(mode="json"))), "x"
    )
    assert again == cfg
