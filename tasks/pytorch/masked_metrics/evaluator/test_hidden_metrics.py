import torch
from metrics import box_iou, masked_l1_loss, masked_mean

torch.manual_seed(0)


def loop_masked_mean(v, m):
    out = torch.zeros(v.shape[0], *v.shape[2:], dtype=v.dtype)
    for b in range(v.shape[0]):
        idx = m[b].nonzero().flatten()
        if len(idx):
            out[b] = v[b, idx].mean(dim=0)
    return out


def test_masked_mean_matches_loop_3d() -> None:
    v = torch.randn(4, 7, 3)
    m = torch.rand(4, 7) > 0.4
    m[2] = False
    assert torch.allclose(masked_mean(v, m), loop_masked_mean(v, m), atol=1e-6)


def test_masked_mean_2d_and_4d() -> None:
    v2 = torch.randn(3, 5)
    m = torch.rand(3, 5) > 0.5
    assert torch.allclose(masked_mean(v2, m), loop_masked_mean(v2, m), atol=1e-6)
    v4 = torch.randn(3, 5, 2, 2)
    assert torch.allclose(masked_mean(v4, m), loop_masked_mean(v4, m), atol=1e-6)


def test_masked_l1_value() -> None:
    p = torch.randn(2, 4, 3)
    t = torch.randn(2, 4, 3)
    m = torch.tensor([[True, True, False, False], [True, False, False, False]])
    expected = (p - t).abs()[m].mean()
    assert torch.allclose(masked_l1_loss(p, t, m), expected, atol=1e-6)


def test_iou_known_values() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0]])
    expected = torch.tensor([[1 / 7, 0.0], [0.0, 0.0]])
    assert torch.allclose(box_iou(a, b), expected, atol=1e-6)


def test_iou_empty() -> None:
    assert box_iou(torch.zeros(0, 4), torch.zeros(3, 4)).shape == (0, 3)
