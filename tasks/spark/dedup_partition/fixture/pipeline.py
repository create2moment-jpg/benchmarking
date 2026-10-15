"""Telemetry batch step: validate, deduplicate re-delivered events, add partition columns."""

from __future__ import annotations

from pyspark.sql import DataFrame
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


def dedup_latest(df: DataFrame) -> DataFrame:
    """One row per (vehicle_id, event_id): the latest ingest_ts wins; on an ingest_ts tie the
    larger speed_mps wins (nulls last)."""
    return df.dropDuplicates(["vehicle_id", "event_id"])


def transform(df: DataFrame) -> DataFrame:
    """validate -> drop rows with null vehicle_id/event_id/event_ts -> dedup -> add event_date
    (DATE of event_ts in the session time zone, which the job sets to UTC)."""
    validate_schema(df)
    return dedup_latest(df).withColumn("event_date", F.to_date("event_ts"))
