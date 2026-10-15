"""Merge closed time intervals (sensor dropout windows)."""

from __future__ import annotations

import math

Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping or touching closed intervals.

    Returns a new list sorted by start. Raises ValueError for NaN bounds or start > end.
    The input list is not modified.
    """
    for start, end in intervals:
        if math.isnan(start) or math.isnan(end):
            raise ValueError(f"interval contains NaN: {(start, end)}")
        if start > end:
            raise ValueError(f"interval start > end: {(start, end)}")
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
