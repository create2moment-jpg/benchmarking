import math

import pytest
from tracker import TrackManager


def test_out_of_order_rejected_state_unchanged() -> None:
    tm = TrackManager()
    tm.step(1.0, [(0.0, 0.0)])
    with pytest.raises(ValueError):
        tm.step(0.5, [(5.0, 5.0)])
    assert [t.id for t in tm.tracks] == [1]


def test_duplicate_timestamp_rejected() -> None:
    tm = TrackManager()
    tm.step(1.0, [])
    with pytest.raises(ValueError):
        tm.step(1.0, [])


def test_nan_detection_ignored() -> None:
    tm = TrackManager()
    tracks = tm.step(0.0, [(math.nan, 0.0), (1.0, 1.0)])
    assert [t.position for t in tracks] == [(1.0, 1.0)]


@pytest.mark.parametrize("kwargs", [{"confirm_hits": 0}, {"max_misses": -1}, {"gate": 0.0}, {"gate": math.inf}])
def test_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        TrackManager(**kwargs)
