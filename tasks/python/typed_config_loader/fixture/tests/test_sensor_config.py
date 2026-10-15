from sensor_config import Extrinsics, SensorConfig, load_sensor_configs


def test_valid_minimal() -> None:
    raw = {"sensors": [{"name": "front", "kind": "radar", "rate_hz": 20,
                        "extrinsics": {"x": 3.5, "y": 0.0, "z": 0.5}}]}
    assert load_sensor_configs(raw) == [
        SensorConfig("front", "radar", 20.0, Extrinsics(3.5, 0.0, 0.5, 0.0))
    ]
