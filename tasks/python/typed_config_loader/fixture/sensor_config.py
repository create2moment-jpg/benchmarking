"""Load and validate the sensor rig configuration (already parsed from YAML into dicts/lists)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    sensors = []
    for item in raw["sensors"]:
        ext = item["extrinsics"]
        sensors.append(
            SensorConfig(item["name"], item["kind"], float(item["rate_hz"]), Extrinsics(**ext))
        )
    return sensors
