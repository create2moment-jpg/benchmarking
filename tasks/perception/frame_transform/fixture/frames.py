"""SE(2) coordinate-frame utilities for vehicle <-> world transforms (NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """Pose of frame B expressed in frame A: translation (x, y) and heading yaw (radians)."""

    x: float
    y: float
    yaw: float


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def transform_points(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Map (N, 2) points from frame B into frame A using ``pose`` (B in A)."""
    return points @ _rot(pose.yaw) + np.array([pose.x, pose.y])


def compose(a_b: Pose2D, b_c: Pose2D) -> Pose2D:
    """Pose of C in A given B in A and C in B."""
    t = transform_points(np.array([[b_c.x, b_c.y]]), a_b)[0]
    return Pose2D(float(t[0]), float(t[1]), a_b.yaw + b_c.yaw)


def inverse(a_b: Pose2D) -> Pose2D:
    """Pose of A in B."""
    return Pose2D(-a_b.x, -a_b.y, -a_b.yaw)


def interpolate_pose(p0: Pose2D, t0: float, p1: Pose2D, t1: float, t: float) -> Pose2D:
    """Linearly interpolate a pose at time t in [t0, t1] (ego-motion compensation)."""
    w = (t - t0) / (t1 - t0)
    return Pose2D(
        p0.x + w * (p1.x - p0.x), p0.y + w * (p1.y - p0.y), p0.yaw + w * (p1.yaw - p0.yaw)
    )
