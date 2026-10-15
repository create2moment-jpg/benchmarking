import pytest
from sync import match_nearest


def test_middle_frames_match() -> None:
    assert match_nearest([1.01, 2.02], [1.0, 2.0, 3.0], tolerance=0.05) == [0, 1]


def test_out_of_tolerance() -> None:
    assert match_nearest([1.5], [1.0, 2.0, 3.0], tolerance=0.1) == [None]


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        match_nearest([1.0], [1.0], tolerance=-1.0)


def test_last_target_can_match_regression() -> None:
    assert match_nearest([10.04], [9.0, 10.0], tolerance=0.05) == [1]


def test_unsorted_target_rejected() -> None:
    with pytest.raises(ValueError):
        match_nearest([1.0], [2.0, 1.0], tolerance=0.5)
