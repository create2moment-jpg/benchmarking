from intervals import merge_intervals


def test_empty() -> None:
    assert merge_intervals([]) == []


def test_overlapping_sorted() -> None:
    assert merge_intervals([(0.0, 2.0), (1.0, 3.0)]) == [(0.0, 3.0)]


def test_disjoint() -> None:
    assert merge_intervals([(0.0, 1.0), (5.0, 6.0)]) == [(0.0, 1.0), (5.0, 6.0)]
