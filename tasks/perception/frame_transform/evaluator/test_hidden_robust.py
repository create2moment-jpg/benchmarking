import math

import numpy as np
import pytest
from frames import Pose2D, interpolate_pose, transform_points


def test_bad_shape_rejected() -> None:
    with pytest.raises(ValueError):
        transform_points(np.zeros((3, 3)), Pose2D(0, 0, 0))
    with pytest.raises(ValueError):
        transform_points(np.zeros(2), Pose2D(0, 0, 0))


def test_input_not_mutated() -> None:
    pts = np.array([[1.0, 2.0]])
    transform_points(pts, Pose2D(1.0, 1.0, 1.0))
    assert np.array_equal(pts, [[1.0, 2.0]])


def test_no_extrapolation() -> None:
    p = Pose2D(0, 0, 0)
    with pytest.raises(ValueError):
        interpolate_pose(p, 0.0, p, 1.0, 1.5)


def test_degenerate_interval() -> None:
    p = Pose2D(0, 0, 0)
    with pytest.raises(ValueError):
        interpolate_pose(p, 1.0, p, 1.0, 1.0)


def test_yaw_output_wrapped() -> None:
    p0 = Pose2D(0, 0, math.radians(179))
    p1 = Pose2D(0, 0, math.radians(-179))
    yaw = interpolate_pose(p0, 0.0, p1, 1.0, 1.0).yaw
    assert -math.pi <= yaw < math.pi
