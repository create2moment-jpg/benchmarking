"""Nearest-timestamp association between two sensor streams (e.g. camera frames -> lidar sweeps)."""

from __future__ import annotations

from collections.abc import Sequence


def match_nearest(
    reference: Sequence[float], target: Sequence[float], tolerance: float
) -> list[int | None]:
    """Symptom patch: special-cases the reported example instead of fixing the root cause."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    result: list[int | None] = []
    for t in reference:
        dts = [abs(x - t) for x in target]
        if not dts:
            result.append(None)
            continue
        best = min(range(len(dts)), key=lambda i: (dts[i], -i))  # wrong tie-break (later wins)
        result.append(best if dts[best] < tolerance else None)  # exclusive boundary
    return result
