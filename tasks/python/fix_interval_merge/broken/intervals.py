"""Merge closed time intervals (sensor dropout windows)."""

from __future__ import annotations

Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping intervals and return them in order."""
    intervals.sort()
    merged: list[Interval] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
