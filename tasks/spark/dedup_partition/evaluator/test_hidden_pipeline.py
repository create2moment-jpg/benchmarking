import datetime as dt

import pytest
from pipeline import dedup_latest, transform, validate_schema

T = dt.datetime
SCHEMA = "vehicle_id string, event_id string, event_ts timestamp, ingest_ts timestamp, speed_mps double"


def rows(df):
    return sorted((r.vehicle_id, r.event_id, r.speed_mps) for r in df.collect())


def test_latest_ingest_wins(spark) -> None:
    df = spark.createDataFrame([
        ("v1", "e1", T(2026, 1, 1, 1), T(2026, 1, 1, 1, 0, 1), 1.0),
        ("v1", "e1", T(2026, 1, 1, 1), T(2026, 1, 1, 1, 0, 9), 2.0),
        ("v1", "e1", T(2026, 1, 1, 1), T(2026, 1, 1, 1, 0, 5), 3.0),
        ("v2", "e1", T(2026, 1, 1, 1), T(2026, 1, 1, 1, 0, 1), 4.0),
    ], SCHEMA)
    assert rows(dedup_latest(df)) == [("v1", "e1", 2.0), ("v2", "e1", 4.0)]


def test_tie_breaks_on_speed_nulls_last(spark) -> None:
    df = spark.createDataFrame([
        ("v1", "e1", T(2026, 1, 1), T(2026, 1, 1, 0, 1), None),
        ("v1", "e1", T(2026, 1, 1), T(2026, 1, 1, 0, 1), 7.0),
        ("v1", "e1", T(2026, 1, 1), T(2026, 1, 1, 0, 1), 3.0),
    ], SCHEMA)
    assert rows(dedup_latest(df)) == [("v1", "e1", 7.0)]


def test_transform_end_to_end(spark) -> None:
    df = spark.createDataFrame([
        ("v1", "e1", T(2026, 1, 1, 23, 59), T(2026, 1, 2, 0, 1), 1.0),
        ("v1", "e1", T(2026, 1, 1, 23, 59), T(2026, 1, 2, 0, 2), 1.5),
        ("v1", "e2", T(2026, 1, 2, 0, 0), T(2026, 1, 2, 0, 1), 2.0),
        (None, "e3", T(2026, 1, 2, 0, 0), T(2026, 1, 2, 0, 1), 9.0),
        ("v2", "e4", None, T(2026, 1, 2, 0, 1), 9.0),
    ], SCHEMA)
    out = sorted((r.vehicle_id, r.event_id, r.speed_mps, r.event_date) for r in transform(df).collect())
    assert out == [
        ("v1", "e1", 1.5, dt.date(2026, 1, 1)),
        ("v1", "e2", 2.0, dt.date(2026, 1, 2)),
    ]
    assert set(transform(df).columns) == {"vehicle_id", "event_id", "event_ts", "ingest_ts", "speed_mps", "event_date"}
