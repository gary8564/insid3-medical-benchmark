"""Tight GT box. ``max_shift=0`` — do not randomly expand."""

from __future__ import annotations

import numpy as np


def mask2D_to_bbox(gt2d: np.ndarray, max_shift: int = 0) -> np.ndarray | None:
    """xyxy box in original GT pixels. ``None`` if the mask is empty."""
    mask = np.asarray(gt2d) > 0
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {mask.shape}")
    if not mask.any():
        return None
    y_indices, x_indices = np.where(mask)
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    height, width = mask.shape
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    if max_shift > 0:
        shift = int(np.random.randint(0, max_shift + 1))
        x_min = max(0, x_min - shift)
        x_max = min(width - 1, x_max + shift)
        y_min = max(0, y_min - shift)
        y_max = min(height - 1, y_max + shift)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)
