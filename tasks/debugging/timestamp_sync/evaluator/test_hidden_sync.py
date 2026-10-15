import pytest
from sync import match_nearest


@pytest.mark.parametrize(
    ("ref", "target", "tol", "expected"),
    [
        ([10.04], [9.0, 10.0], 0.05, [1]),
        ([0.0, 5.0, 9.99], [0.0, 5.0, 10.0], 0.02, [0, 1, 2]),
        ([2.5], [2.0, 3.0], 0.5, [0]),          # exact tie -> earlier index
        ([1.25], [1.0], 0.25, [0]),               # boundary inclusive (float-safe values)
        ([3.0], [1.0, 2.0], 0.5, [None]),
        ([-1.0], [0.0, 1.0], 0.5, [None]),
        ([], [1.0], 0.1, []),
        ([1.0, 2.0], [], 0.1, [None, None]),
        ([1.0], [1.0, 1.0], 0.0, [0]),          # duplicate timestamps -> earliest
    ],
)
def test_matching(ref, target, tol, expected) -> None:
    assert match_nearest(ref, target, tol) == expected
