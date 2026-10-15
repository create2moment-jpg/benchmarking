import pytest
import torch
from metrics import box_iou, masked_l1_loss, masked_mean


def test_fully_masked_row_is_zero_not_nan() -> None:
    v = torch.randn(2, 3, 4)
    m = torch.tensor([[False, False, False], [True, False, True]])
    out = masked_mean(v, m)
    assert torch.isfinite(out).all() and torch.all(out[0] == 0)


def test_all_masked_loss_zero_with_finite_grad() -> None:
    p = torch.randn(2, 3, 2, requires_grad=True)
    t = torch.randn(2, 3, 2)
    loss = masked_l1_loss(p, t, torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(p.grad).all()


def test_masked_values_do_not_leak_nan() -> None:
    v = torch.tensor([[[1.0], [float("nan")]]])
    m = torch.tensor([[True, False]])
    assert torch.allclose(masked_mean(v, m), torch.tensor([[1.0]]))


def test_degenerate_boxes_no_nan() -> None:
    a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert torch.equal(box_iou(a, a), torch.zeros(1, 1))


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        masked_mean(torch.randn(2, 3), torch.ones(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        box_iou(torch.zeros(2, 3), torch.zeros(2, 4))
