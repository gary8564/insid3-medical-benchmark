"""Binary mask helpers. Score only after the prediction matches the GT PNG shape."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_binary_mask(path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)


def as_binary_mask(pred) -> np.ndarray:
    array = np.asarray(pred.detach().cpu() if hasattr(pred, "detach") else pred)
    binary = (array != 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2D predicted mask, got {binary.shape}")
    if set(np.unique(binary).tolist()) - {0, 1}:
        raise ValueError("predicted mask values must be in {0, 1}")
    return binary


def resize_pred_to_gt(pred: np.ndarray, gt_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to the original GT PNG shape. Never score on the model canvas."""
    binary = as_binary_mask(pred)
    height, width = gt_shape
    if binary.shape == (height, width):
        return binary
    resized = Image.fromarray(binary * 255, mode="L").resize((width, height), resample=Image.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)
