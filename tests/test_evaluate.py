from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import binary_dice, binary_iou, mean_metrics


def test_perfect_overlap():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert binary_iou(mask, mask) == 1.0
    assert binary_dice(mask, mask) == 1.0


def test_no_overlap():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert binary_iou(pred, gt) == 0.0
    assert binary_dice(pred, gt) == 0.0


def test_known_partial_overlap():
    # intersection 1, union 3 → IoU 1/3; Dice 2/4 = 0.5
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert binary_iou(pred, gt) == pytest.approx(1 / 3)
    assert binary_dice(pred, gt) == pytest.approx(0.5)


def test_both_empty_is_perfect_agreement():
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert binary_iou(empty, empty) == 1.0
    assert binary_dice(empty, empty) == 1.0


def test_mean_metrics():
    ones = np.ones((2, 2), dtype=np.uint8)
    zeros = np.zeros((2, 2), dtype=np.uint8)
    summary = mean_metrics([(ones, ones), (ones, zeros)])
    assert summary["n"] == 2
    assert summary["mIoU"] == pytest.approx(0.5)
    assert summary["Dice"] == pytest.approx(0.5)


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_iou(np.zeros((2, 2)), np.zeros((3, 3)))
