"""Binary segmentation metrics (mIoU / Dice)."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {array.shape}")
    return array != 0


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    if pred_b.shape != gt_b.shape:
        raise ValueError(f"shape mismatch: pred {pred_b.shape} vs gt {gt_b.shape}")
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = _as_bool(pred)
    gt_b = _as_bool(gt)
    if pred_b.shape != gt_b.shape:
        raise ValueError(f"shape mismatch: pred {pred_b.shape} vs gt {gt_b.shape}")
    intersection = np.logical_and(pred_b, gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def mean_metrics(pred_gt_pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    ious = [binary_iou(pred, gt) for pred, gt in pred_gt_pairs]
    dices = [binary_dice(pred, gt) for pred, gt in pred_gt_pairs]
    return {
        "mIoU": float(np.mean(ious)) if ious else float("nan"),
        "Dice": float(np.mean(dices)) if dices else float("nan"),
        "n": float(len(pred_gt_pairs)),
    }
