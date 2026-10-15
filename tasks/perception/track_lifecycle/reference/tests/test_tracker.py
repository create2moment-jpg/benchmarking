from tracker import TrackManager, TrackState


def test_new_detection_creates_tentative_track() -> None:
    tm = TrackManager()
    tracks = tm.step(0.0, [(0.0, 0.0)])
    assert len(tracks) == 1
    assert tracks[0].state is TrackState.TENTATIVE
    assert tracks[0].id == 1


def test_confirmation_after_hits() -> None:
    tm = TrackManager(confirm_hits=3)
    for i in range(3):
        tracks = tm.step(float(i), [(0.1 * i, 0.0)])
    assert tracks[0].state is TrackState.CONFIRMED


def test_out_of_order_timestamp_rejected() -> None:
    import pytest

    tm = TrackManager()
    tm.step(1.0, [])
    with pytest.raises(ValueError):
        tm.step(0.0, [])
