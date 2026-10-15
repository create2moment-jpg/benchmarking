"""Masked tensor utilities for padded detection/tracking batches (CPU, no training)."""

from __future__ import annotations

import torch


def _check_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor")
    if values.dim() < 2 or mask.shape != values.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} must equal values.shape[:2]")
    return mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the time axis (dim 1) of ``values`` [B, T, ...] using boolean ``mask`` [B, T].

    Returns [B, ...]. A batch row with no valid steps returns 0 (never NaN).
    """
    _check_mask(values, mask)
    m = mask.reshape(*mask.shape, *([1] * (values.dim() - 2)))
    # where() rather than multiply: NaN * 0 is NaN, so padded garbage must be replaced.
    total = torch.where(m, values, torch.zeros((), dtype=values.dtype)).sum(dim=1)
    count = m.to(values.dtype).sum(dim=1)
    return total / count.clamp(min=1)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid elements of [B, T, D] tensors (mask [B, T]).

    The mean is over valid (b, t, d) elements. With no valid element the loss is 0 and its
    gradient is finite (zero).
    """
    if pred.shape != target.shape or pred.dim() != 3:
        raise ValueError("pred and target must have the same [B, T, D] shape")
    _check_mask(pred, mask)
    m = mask[..., None].expand_as(pred)
    err = torch.where(m, (pred - target).abs(), torch.zeros((), dtype=pred.dtype))
    count = m.to(pred.dtype).sum()
    return err.sum() / count.clamp(min=1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of axis-aligned boxes in (x1, y1, x2, y2) format: [N, 4] x [M, 4] -> [N, M].

    Degenerate (zero-area) boxes give IoU 0, never NaN.
    """
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != 4 or b.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=torch.finfo(a.dtype).tiny), 0.0)
