import torch
from metrics import box_iou


def test_identical_boxes_iou_one() -> None:
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    assert torch.allclose(box_iou(a, a), torch.ones(1, 1))
