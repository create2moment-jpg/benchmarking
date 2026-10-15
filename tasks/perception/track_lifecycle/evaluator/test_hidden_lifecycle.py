import pytest
from tracker import TrackManager, TrackState


def confirmed(tm: TrackManager, start: float = 0.0, pos=(0.0, 0.0)) -> float:
    t = start
    for _ in range(tm.confirm_hits):
        tm.step(t, [pos])
        t += 1.0
    return t


def test_tentative_deleted_on_first_miss() -> None:
    tm = TrackManager(confirm_hits=3)
    tm.step(0.0, [(0.0, 0.0)])
    assert tm.step(1.0, []) == []


def test_confirmed_survives_max_misses_then_deleted() -> None:
    tm = TrackManager(confirm_hits=2, max_misses=2)
    t = confirmed(tm)
    assert len(tm.step(t, [])) == 1
    assert len(tm.step(t + 1, [])) == 1
    assert tm.step(t + 2, []) == []


def test_miss_counter_resets_on_hit() -> None:
    tm = TrackManager(confirm_hits=2, max_misses=1)
    t = confirmed(tm)
    tm.step(t, [])
    tracks = tm.step(t + 1, [(0.1, 0.0)])
    assert tracks[0].misses == 0
    assert len(tm.step(t + 2, [])) == 1


def test_ids_never_reused() -> None:
    tm = TrackManager(confirm_hits=3)
    tm.step(0.0, [(0.0, 0.0), (10.0, 10.0)])
    tm.step(1.0, [(10.0, 10.0)])            # track 1 (tentative) deleted
    tracks = tm.step(2.0, [(10.0, 10.0), (50.0, 50.0)])
    assert sorted(t.id for t in tracks) == [2, 3]


def test_global_greedy_association() -> None:
    tm = TrackManager(confirm_hits=1, gate=5.0)
    tm.step(0.0, [(0.0, 0.0), (3.0, 0.0)])  # tracks 1 @0, 2 @3
    # det A at 2.0 is closest to track 2 (d=1), det B at -1 closest to track 1 (d=1).
    # Iterating tracks in order and grabbing nearest would steal A for track 1 (d=2).
    tracks = tm.step(1.0, [(2.0, 0.0), (-1.0, 0.0)])
    pos = {t.id: t.position for t in tracks}
    assert pos == {1: (-1.0, 0.0), 2: (2.0, 0.0)}


def test_gate_respected() -> None:
    tm = TrackManager(confirm_hits=1, gate=1.0)
    tm.step(0.0, [(0.0, 0.0)])
    tracks = tm.step(1.0, [(1.5, 0.0)])
    assert sorted(t.id for t in tracks) == [1, 2]


def test_confirm_hits_one_confirms_immediately() -> None:
    tm = TrackManager(confirm_hits=1)
    assert tm.step(0.0, [(0.0, 0.0)])[0].state is TrackState.CONFIRMED


def test_last_update_ts() -> None:
    tm = TrackManager(confirm_hits=2, max_misses=3)
    tm.step(0.0, [(0.0, 0.0)])
    tm.step(1.0, [(0.0, 0.0)])
    tracks = tm.step(2.0, [])
    assert tracks[0].last_update_ts == 1.0
