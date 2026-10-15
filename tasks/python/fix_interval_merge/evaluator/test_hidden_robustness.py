import math

import pytest
from intervals import merge_intervals


def test_rejects_reversed() -> None:
    with pytest.raises(ValueError):
        merge_intervals([(2.0, 1.0)])


def test_rejects_nan() -> None:
    with pytest.raises(ValueError):
        merge_intervals([(math.nan, 1.0)])


def test_does_not_mutate_input() -> None:
    data = [(5.0, 6.0), (0.0, 1.0)]
    merge_intervals(data)
    assert data == [(5.0, 6.0), (0.0, 1.0)]


def test_returns_new_list() -> None:
    data = [(0.0, 1.0)]
    assert merge_intervals(data) is not data
