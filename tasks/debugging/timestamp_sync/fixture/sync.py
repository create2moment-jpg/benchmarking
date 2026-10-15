"""Nearest-timestamp association between two sensor streams (e.g. camera frames -> lidar sweeps)."""

from __future__ import annotations

import bisect
from collections.abc import Sequence


def match_nearest(
    reference: Sequence[float], target: Sequence[float], tolerance: float
) -> list[int | None]:
    """For each reference timestamp return the index of the nearest target timestamp.

    Contract:
    - ``target`` must be sorted ascending; otherwise ``ValueError``.
    - ``tolerance`` must be >= 0; otherwise ``ValueError``.
    - A match requires ``abs(target[i] - ref) <= tolerance``; otherwise the result is ``None``.
    - On an exact tie between two candidates, the EARLIER target index wins.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    result: list[int | None] = []
    for t in reference:
        idx = bisect.bisect_left(target, t)
        best: int | None = None
        best_dt = tolerance
        for cand in (idx - 1, idx):
            if 0 <= cand < len(target) - 1:
                dt = abs(target[cand] - t)
                if dt <= best_dt and (best is None or dt < abs(target[best] - t)):
                    best, best_dt = cand, dt
        result.append(best)
    return result
