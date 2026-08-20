from __future__ import annotations

import numpy as np

from src.methods.medsam2.box import mask2D_to_bbox


def test_tight_box_max_shift_zero():
    gt = np.zeros((10, 12), dtype=np.uint8)
    gt[2:5, 3:8] = 1
    box = mask2D_to_bbox(gt, max_shift=0)
    assert box.tolist() == [3, 2, 7, 4]


def test_empty_gt_returns_none():
    assert mask2D_to_bbox(np.zeros((4, 4), dtype=np.uint8), max_shift=0) is None


def test_max_shift_zero_is_deterministic():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[1:4, 2:6] = 1
    first = mask2D_to_bbox(gt, max_shift=0)
    second = mask2D_to_bbox(gt, max_shift=0)
    assert first.tolist() == second.tolist() == [2, 1, 5, 3]
