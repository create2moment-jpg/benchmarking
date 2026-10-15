"""SE(2) coordinate-frame utilities for vehicle <-> world transforms (NumPy)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """Pose of frame B expressed in frame A: translation (x, y) and heading yaw (radians)."""

    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform_points(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Map (N, 2) points from frame B into frame A using ``pose`` (B in A). Returns a new array."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    return pts @ _rot(pose.yaw).T + np.array([pose.x, pose.y])


def compose(a_b: Pose2D, b_c: Pose2D) -> Pose2D:
    """Pose of C in A given B in A and C in B."""
    t = transform_points(np.array([[b_c.x, b_c.y]]), a_b)[0]
    return Pose2D(float(t[0]), float(t[1]), wrap_angle(a_b.yaw + b_c.yaw))


def inverse(a_b: Pose2D) -> Pose2D:
    """Pose of A in B."""
    r_t = _rot(a_b.yaw).T
    t = -(r_t @ np.array([a_b.x, a_b.y]))
    return Pose2D(float(t[0]), float(t[1]), wrap_angle(-a_b.yaw))


def interpolate_pose(p0: Pose2D, t0: float, p1: Pose2D, t1: float, t: float) -> Pose2D:
    """Interpolate a pose at time t in [t0, t1] along the shortest yaw arc."""
    if not t1 > t0:
        raise ValueError("t1 must be greater than t0")
    if not t0 <= t <= t1:
        raise ValueError("t must lie within [t0, t1] (no extrapolation)")
    w = (t - t0) / (t1 - t0)
    dyaw = wrap_angle(p1.yaw - p0.yaw)
    return Pose2D(
        p0.x + w * (p1.x - p0.x), p0.y + w * (p1.y - p0.y), wrap_angle(p0.yaw + w * dyaw)
    )
