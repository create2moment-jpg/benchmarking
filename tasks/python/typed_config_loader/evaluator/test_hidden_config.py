import pytest
from sensor_config import ConfigError, Extrinsics, SensorConfig, load_sensor_configs


def sensor(**over):
    base = {"name": "front", "kind": "radar", "rate_hz": 20,
            "extrinsics": {"x": 1.0, "y": 2.0, "z": 0.5, "yaw_deg": 90}}
    base.update(over)
    return base


def paths(exc: ConfigError) -> set[str]:
    return {p for p, _ in exc.errors}


def test_valid_multiple_in_order() -> None:
    raw = {"sensors": [sensor(), sensor(name="roof", kind="lidar", rate_hz=10.5,
                                       extrinsics={"x": 0, "y": 0, "z": 2})]}
    assert load_sensor_configs(raw) == [
        SensorConfig("front", "radar", 20.0, Extrinsics(1.0, 2.0, 0.5, 90.0)),
        SensorConfig("roof", "lidar", 10.5, Extrinsics(0.0, 0.0, 2.0, 0.0)),
    ]


@pytest.mark.parametrize(
    ("bad", "path"),
    [
        (sensor(rate_hz=0), "sensors[0].rate_hz"),
        (sensor(rate_hz=1000.5), "sensors[0].rate_hz"),
        (sensor(rate_hz=True), "sensors[0].rate_hz"),
        (sensor(rate_hz="20"), "sensors[0].rate_hz"),
        (sensor(kind="sonar"), "sensors[0].kind"),
        (sensor(name=""), "sensors[0].name"),
        (sensor(extrinsics={"x": 1, "y": 2}), "sensors[0].extrinsics.z"),
        (sensor(extrinsics={"x": 1, "y": 2, "z": 3, "yaw_deg": 270}), "sensors[0].extrinsics.yaw_deg"),
        (sensor(extrinsics={"x": 1, "y": 2, "z": 3, "pitch": 1}), "sensors[0].extrinsics.pitch"),
        (sensor(fps=30), "sensors[0].fps"),
    ],
)
def test_single_error_path(bad, path) -> None:
    with pytest.raises(ConfigError) as info:
        load_sensor_configs({"sensors": [bad]})
    assert path in paths(info.value)


def test_duplicate_names() -> None:
    with pytest.raises(ConfigError) as info:
        load_sensor_configs({"sensors": [sensor(), sensor()]})
    assert "sensors[1].name" in paths(info.value)


def test_all_errors_reported() -> None:
    raw = {"sensors": [sensor(rate_hz=-1), sensor(name="b", kind="x")]}
    with pytest.raises(ConfigError) as info:
        load_sensor_configs(raw)
    assert {"sensors[0].rate_hz", "sensors[1].kind"} <= paths(info.value)
