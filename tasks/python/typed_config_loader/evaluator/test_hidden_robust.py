import pytest
from sensor_config import ConfigError, load_sensor_configs

GARBAGE = [
    None, [], "sensors", 42, {}, {"sensors": None}, {"sensors": []}, {"sensors": "abc"},
    {"sensors": [None]}, {"sensors": [42]}, {"sensors": [{}]},
    {"sensors": [{"name": "a", "kind": "radar", "rate_hz": 1, "extrinsics": None}]},
    {"sensors": [{"name": "a", "kind": "radar", "rate_hz": 1, "extrinsics": [1, 2, 3]}]},
    {"sensors": [{"name": ["a"], "kind": {"r": 1}, "rate_hz": None, "extrinsics": {"x": "1"}}]},
    {"sensors": [], "extra": 1},
]


@pytest.mark.parametrize("raw", GARBAGE)
def test_garbage_raises_config_error_only(raw) -> None:
    with pytest.raises(ConfigError):
        load_sensor_configs(raw)


def test_nan_rate_rejected() -> None:
    raw = {"sensors": [{"name": "a", "kind": "radar", "rate_hz": float("nan"),
                        "extrinsics": {"x": 0, "y": 0, "z": 0}}]}
    with pytest.raises(ConfigError):
        load_sensor_configs(raw)
