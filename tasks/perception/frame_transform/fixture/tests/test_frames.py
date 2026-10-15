import numpy as np
from frames import Pose2D, transform_points


def test_pure_translation() -> None:
    pts = np.array([[1.0, 2.0]])
    out = transform_points(pts, Pose2D(10.0, -1.0, 0.0))
    assert np.allclose(out, [[11.0, 1.0]])
