"""Masked tensor utilities for padded detection/tracking batches (CPU, no training)."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the time axis (dim 1) of ``values`` [B, T, ...] using boolean ``mask`` [B, T].

    Returns [B, ...]. A batch row with no valid steps returns 0 (never NaN).
    """
    return (values * mask).sum(dim=1) / values.shape[1]


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid elements of [B, T, D] tensors (mask [B, T]).

    The mean is over valid (b, t, d) elements. With no valid element the loss is 0 and its
    gradient is finite (zero).
    """
    return (pred - target).abs().mean()


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of axis-aligned boxes in (x1, y1, x2, y2) format: [N, 4] x [M, 4] -> [N, M].

    Degenerate (zero-area) boxes give IoU 0, never NaN.
    """
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).prod(dim=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)
