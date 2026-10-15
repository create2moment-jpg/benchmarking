import pytest
from sync import match_nearest


def test_unsorted_target_raises() -> None:
    with pytest.raises(ValueError):
        match_nearest([1.0], [3.0, 1.0, 2.0], 0.5)


def test_negative_tolerance_raises() -> None:
    with pytest.raises(ValueError):
        match_nearest([1.0], [1.0], -0.1)


def test_accepts_tuples() -> None:
    assert match_nearest((1.0,), (0.9, 1.05), 0.1) == [1]
