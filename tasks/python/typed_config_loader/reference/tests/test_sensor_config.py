from sensor_config import Extrinsics, SensorConfig, load_sensor_configs


def test_valid_minimal() -> None:
    raw = {"sensors": [{"name": "front", "kind": "radar", "rate_hz": 20,
                        "extrinsics": {"x": 3.5, "y": 0.0, "z": 0.5}}]}
    assert load_sensor_configs(raw) == [
        SensorConfig("front", "radar", 20.0, Extrinsics(3.5, 0.0, 0.5, 0.0))
    ]


def test_bool_rate_rejected() -> None:
    import pytest
    from sensor_config import ConfigError

    raw = {"sensors": [{"name": "a", "kind": "radar", "rate_hz": True,
                        "extrinsics": {"x": 0, "y": 0, "z": 0}}]}
    with pytest.raises(ConfigError) as info:
        load_sensor_configs(raw)
    assert ("sensors[0].rate_hz", "must be a number") in info.value.errors


def test_missing_sensors_key() -> None:
    import pytest
    from sensor_config import ConfigError

    with pytest.raises(ConfigError):
        load_sensor_configs({})
