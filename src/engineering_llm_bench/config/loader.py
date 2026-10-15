"""Load and validate YAML configuration with actionable error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from engineering_llm_bench.config.models import BenchmarkConfig, ModelsFile

T = TypeVar("T", bound=BaseModel)

MAX_CONFIG_BYTES = 1_000_000


class ConfigError(Exception):
    """A configuration file is missing, unparsable or invalid. Message is user-facing."""


def read_yaml(path: Path) -> Any:
    """Parse a YAML file with ``safe_load`` (no arbitrary object construction)."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigError(f"config file too large (> {MAX_CONFIG_BYTES} bytes): {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def format_validation_error(exc: ValidationError, source: str) -> str:
    """Render pydantic errors as ``source: field.path: message`` lines."""
    lines = [f"{source}: {exc.error_count()} validation error(s)"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_model(model: type[T], data: Any, source: str) -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at top level")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, source)) from exc


def load_models(path: Path) -> ModelsFile:
    return parse_model(ModelsFile, read_yaml(path), str(path))


def load_benchmark(path: Path) -> BenchmarkConfig:
    return parse_model(BenchmarkConfig, read_yaml(path), str(path))
