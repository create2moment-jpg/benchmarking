import pytest
from pipeline import transform, validate_schema


def test_missing_and_wrong_type_columns_reported(spark) -> None:
    df = spark.createDataFrame([("v1", "e1", "2026-01-01", 1)],
                               "vehicle_id string, event_id string, event_ts string, speed_mps int")
    with pytest.raises(ValueError) as info:
        validate_schema(df)
    msg = str(info.value)
    assert "ingest_ts" in msg and "event_ts" in msg and "speed_mps" in msg


def test_transform_validates(spark) -> None:
    df = spark.createDataFrame([("v1",)], "vehicle_id string")
    with pytest.raises(ValueError):
        transform(df)
