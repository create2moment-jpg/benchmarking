"""Minimal multi-object track manager (2-D positions, nearest-neighbour association)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:
    id: int
    state: TrackState
    position: tuple[float, float]
    hits: int = 1
    misses: int = 0
    last_update_ts: float = 0.0


class TrackManager:
    """See the task description for the full lifecycle contract."""

    def __init__(self, confirm_hits: int = 3, max_misses: int = 2, gate: float = 2.0) -> None:
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.gate = gate
        self.tracks: list[Track] = []

    def step(self, ts: float, detections: list[tuple[float, float]]) -> list[Track]:
        used: set[int] = set()
        for track in self.tracks:
            best, best_d = None, self.gate
            for j, det in enumerate(detections):
                if j in used:
                    continue
                d = math.dist(track.position, det)
                if d <= best_d:
                    best, best_d = j, d
            if best is None:
                track.misses += 1
                if track.misses > self.max_misses:
                    track.state = TrackState.DELETED
            else:
                used.add(best)
                track.position = detections[best]
                track.hits += 1
                track.misses = 0
                track.last_update_ts = ts
                if track.hits >= self.confirm_hits:
                    track.state = TrackState.CONFIRMED
        for j, det in enumerate(detections):
            if j not in used:
                self.tracks.append(
                    Track(len(self.tracks) + 1, TrackState.TENTATIVE, det, last_update_ts=ts)
                )
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        return sorted(self.tracks, key=lambda t: t.id)
