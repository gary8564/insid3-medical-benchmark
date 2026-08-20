from __future__ import annotations

import numpy as np
import pytest

from src.protocol.masks import as_binary_mask, resize_pred_to_gt


def test_as_binary_mask_accepts_bool_and_uint8():
    assert as_binary_mask(np.array([[0, 2], [0, 0]])).tolist() == [[0, 1], [0, 0]]
    assert as_binary_mask(np.array([[False, True], [False, False]])).tolist() == [[0, 1], [0, 0]]


def test_as_binary_mask_rejects_3d():
    with pytest.raises(ValueError, match="2D"):
        as_binary_mask(np.zeros((1, 2, 2)))


def test_resize_pred_to_gt_nearest():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    out = resize_pred_to_gt(pred, (4, 4))
    assert out.shape == (4, 4)
    assert out[0, 0] == 1
    assert out[0, 1] == 1
    assert out[1, 0] == 1
    assert int(out.sum()) == 4
