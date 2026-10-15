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
        if confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        if max_misses < 0:
            raise ValueError("max_misses must be >= 0")
        if not gate > 0 or not math.isfinite(gate):
            raise ValueError("gate must be a positive finite number")
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.gate = gate
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_ts: float | None = None

    def step(self, ts: float, detections: list[tuple[float, float]]) -> list[Track]:
        if not math.isfinite(ts) or (self._last_ts is not None and ts <= self._last_ts):
            raise ValueError("timestamps must be finite and strictly increasing")
        self._last_ts = ts
        valid = [
            (j, d) for j, d in enumerate(detections) if all(math.isfinite(c) for c in d)
        ]
        pairs = sorted(
            (math.dist(t.position, d), t.id, j)
            for t in self.tracks
            for j, d in valid
            if math.dist(t.position, d) <= self.gate
        )
        by_id = {t.id: t for t in self.tracks}
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, tid, j in pairs:
            if tid in matched_tracks or j in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(j)
            track = by_id[tid]
            track.position = detections[j]
            track.hits += 1
            track.misses = 0
            track.last_update_ts = ts
            if track.state is TrackState.TENTATIVE and track.hits >= self.confirm_hits:
                track.state = TrackState.CONFIRMED
        for track in self.tracks:
            if track.id in matched_tracks:
                continue
            track.misses += 1
            if track.state is TrackState.TENTATIVE or track.misses > self.max_misses:
                track.state = TrackState.DELETED
        for j, det in valid:
            if j not in matched_dets:
                state = TrackState.CONFIRMED if self.confirm_hits == 1 else TrackState.TENTATIVE
                self.tracks.append(Track(self._next_id, state, det, last_update_ts=ts))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.state is not TrackState.DELETED]
        return sorted(self.tracks, key=lambda t: t.id)
