"""Load and validate the sensor rig configuration (already parsed from YAML into dicts/lists)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

KINDS = frozenset({"radar", "lidar", "camera"})
_SENSOR_KEYS = frozenset({"name", "kind", "rate_hz", "extrinsics"})
_EXT_KEYS = frozenset({"x", "y", "z", "yaw_deg"})


class ConfigError(Exception):
    """Invalid configuration. ``errors`` lists every problem as (path, message)."""

    def __init__(self, errors: list[tuple[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{path}: {msg}" for path, msg in errors))


@dataclass(frozen=True)
class Extrinsics:
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class SensorConfig:
    name: str
    kind: str
    rate_hz: float
    extrinsics: Extrinsics


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value == value


def _unknown(obj: Mapping[Any, Any], allowed: frozenset[str], path: str) -> list[tuple[str, str]]:
    return [(f"{path}.{k}" if path else str(k), "unknown key") for k in obj if k not in allowed]


def _extrinsics(raw: Any, path: str, errors: list[tuple[str, str]]) -> Extrinsics | None:
    if not isinstance(raw, Mapping):
        errors.append((path, "must be a mapping"))
        return None
    errors.extend(_unknown(raw, _EXT_KEYS, path))
    values: dict[str, float] = {}
    for key in ("x", "y", "z"):
        if key not in raw:
            errors.append((f"{path}.{key}", "is required"))
        elif not _is_number(raw[key]):
            errors.append((f"{path}.{key}", "must be a number"))
        else:
            values[key] = float(raw[key])
    yaw = raw.get("yaw_deg", 0.0)
    if not _is_number(yaw):
        errors.append((f"{path}.yaw_deg", "must be a number"))
    elif not -180.0 <= yaw <= 180.0:
        errors.append((f"{path}.yaw_deg", "must be within [-180, 180]"))
    if len(values) != 3 or not _is_number(yaw) or not -180.0 <= yaw <= 180.0:
        return None
    return Extrinsics(values["x"], values["y"], values["z"], float(yaw))


def load_sensor_configs(raw: Any) -> list[SensorConfig]:
    """Validate ``raw`` and return the sensors in input order.

    Contract (see task description for details):
    - ``raw`` must be a mapping with exactly one key ``sensors``: a non-empty list.
    - each sensor: ``name`` (non-empty str, unique), ``kind`` in {"radar", "lidar", "camera"},
      ``rate_hz`` (int/float, 0 < rate_hz <= 1000; bool is NOT a number),
      ``extrinsics`` mapping with numeric ``x``, ``y``, ``z`` and optional ``yaw_deg``
      (-180 <= yaw_deg <= 180, default 0.0). Unknown keys are errors at every level.
    - on any problem raise ``ConfigError`` listing ALL problems with paths such as
      ``sensors[1].rate_hz`` or ``sensors[0].extrinsics.yaw_deg``; never KeyError/TypeError.
    """
    errors: list[tuple[str, str]] = []
    if not isinstance(raw, Mapping):
        raise ConfigError([("", "configuration must be a mapping")])
    errors.extend(_unknown(raw, frozenset({"sensors"}), ""))
    items = raw.get("sensors")
    if not isinstance(items, list) or not items:
        errors.append(("sensors", "must be a non-empty list"))
        raise ConfigError(errors)
    result: list[SensorConfig] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        path = f"sensors[{i}]"
        if not isinstance(item, Mapping):
            errors.append((path, "must be a mapping"))
            continue
        before = len(errors)
        errors.extend(_unknown(item, _SENSOR_KEYS, path))
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append((f"{path}.name", "must be a non-empty string"))
        elif name in seen:
            errors.append((f"{path}.name", f"duplicate sensor name {name!r}"))
        else:
            seen.add(name)
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            errors.append((f"{path}.kind", f"must be one of {sorted(KINDS)}"))
        rate = item.get("rate_hz")
        if not _is_number(rate):
            errors.append((f"{path}.rate_hz", "must be a number"))
        elif not 0 < rate <= 1000:
            errors.append((f"{path}.rate_hz", "must satisfy 0 < rate_hz <= 1000"))
        if "extrinsics" not in item:
            errors.append((f"{path}.extrinsics", "is required"))
            continue
        ext = _extrinsics(item["extrinsics"], f"{path}.extrinsics", errors)
        if len(errors) == before and ext is not None:
            result.append(SensorConfig(str(name), str(kind), float(rate), ext))
    if errors:
        raise ConfigError(errors)
    return result
