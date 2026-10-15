"""Telemetry batch step: validate, deduplicate re-delivered events, add partition columns."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

REQUIRED_SCHEMA = {
    "vehicle_id": "string",
    "event_id": "string",
    "event_ts": "timestamp",
    "ingest_ts": "timestamp",
    "speed_mps": "double",
}


def validate_schema(df: DataFrame) -> None:
    """Raise ValueError naming every missing column or column with the wrong type."""
    actual = dict(df.dtypes)
    problems = []
    for name, expected in REQUIRED_SCHEMA.items():
        if name not in actual:
            problems.append(f"missing column {name}")
        elif actual[name] != expected:
            problems.append(f"column {name} has type {actual[name]}, expected {expected}")
    if problems:
        raise ValueError("; ".join(problems))


def dedup_latest(df: DataFrame) -> DataFrame:
    """One row per (vehicle_id, event_id): the latest ingest_ts wins; on an ingest_ts tie the
    larger speed_mps wins (nulls last)."""
    window = Window.partitionBy("vehicle_id", "event_id").orderBy(
        F.col("ingest_ts").desc_nulls_last(), F.col("speed_mps").desc_nulls_last()
    )
    return (
        df.withColumn("_rn", F.row_number().over(window)).where(F.col("_rn") == 1).drop("_rn")
    )


def transform(df: DataFrame) -> DataFrame:
    """validate -> drop rows with null vehicle_id/event_id/event_ts -> dedup -> add event_date
    (DATE of event_ts in the session time zone, which the job sets to UTC)."""
    validate_schema(df)
    clean = df.dropna(subset=["vehicle_id", "event_id", "event_ts"])
    return dedup_latest(clean).withColumn("event_date", F.to_date("event_ts"))
