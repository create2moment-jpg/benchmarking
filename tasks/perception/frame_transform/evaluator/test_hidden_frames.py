import math

import numpy as np
from frames import Pose2D, compose, interpolate_pose, inverse, transform_points

RNG = np.random.default_rng(7)


def ang_close(a: float, b: float) -> bool:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) < 1e-9


def test_rotation_direction() -> None:
    out = transform_points(np.array([[1.0, 0.0]]), Pose2D(0.0, 0.0, math.pi / 2))
    assert np.allclose(out, [[0.0, 1.0]])


def test_inverse_roundtrip() -> None:
    for _ in range(20):
        p = Pose2D(*RNG.uniform(-50, 50, 2), RNG.uniform(-math.pi, math.pi))
        pts = RNG.uniform(-10, 10, (5, 2))
        back = transform_points(transform_points(pts, p), inverse(p))
        assert np.allclose(back, pts)


def test_compose_matches_sequential_transform() -> None:
    a_b = Pose2D(2.0, -1.0, 0.7)
    b_c = Pose2D(-3.0, 4.0, -2.1)
    pts = RNG.uniform(-5, 5, (6, 2))
    direct = transform_points(pts, compose(a_b, b_c))
    seq = transform_points(transform_points(pts, b_c), a_b)
    assert np.allclose(direct, seq)


def test_compose_with_inverse_is_identity() -> None:
    p = Pose2D(5.0, 6.0, 3.0)
    ident = compose(p, inverse(p))
    assert abs(ident.x) < 1e-9 and abs(ident.y) < 1e-9 and ang_close(ident.yaw, 0.0)


def test_interpolation_wraps_yaw() -> None:
    p0 = Pose2D(0.0, 0.0, math.radians(170))
    p1 = Pose2D(2.0, 0.0, math.radians(-170))
    mid = interpolate_pose(p0, 0.0, p1, 1.0, 0.5)
    assert abs(mid.x - 1.0) < 1e-12
    assert ang_close(mid.yaw, math.pi)


def test_empty_points() -> None:
    out = transform_points(np.zeros((0, 2)), Pose2D(1.0, 2.0, 0.3))
    assert out.shape == (0, 2)
