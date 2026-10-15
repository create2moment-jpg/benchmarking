# Run: python -m pytest -q tests   (starts a local Spark session; takes a few seconds)
import datetime as dt

import pytest
from pyspark.sql import SparkSession

from pipeline import transform

T = dt.datetime


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.session.timeZone", "UTC").config("spark.ui.enabled", "false")
         .getOrCreate())
    yield s
    s.stop()


def test_transform_adds_event_date(spark) -> None:
    df = spark.createDataFrame(
        [("v1", "e1", T(2026, 1, 1, 10), T(2026, 1, 1, 10, 1), 5.0)],
        "vehicle_id string, event_id string, event_ts timestamp, ingest_ts timestamp, speed_mps double",
    )
    row = transform(df).collect()[0]
    assert row.event_date == dt.date(2026, 1, 1)
