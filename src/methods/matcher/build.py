"""Put ``third_party/Matcher`` on ``sys.path`` and build Matcher (OSS flags)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.data.paths import DINOV2_VITL14_WEIGHTS, MATCHER_ROOT, SAM_VIT_H_WEIGHTS

# NumPy 2 removed the ``np.int`` alias that Matcher's RobustPromptSampler uses.
if not hasattr(np, "int"):
    np.int = np.int64  # type: ignore[attr-defined]


def add_matcher_to_syspath() -> Path:
    if not (MATCHER_ROOT / "matcher" / "Matcher.py").is_file():
        raise FileNotFoundError(
            f"Matcher submodule is missing at {MATCHER_ROOT}. "
            "From the repo root run: git submodule update --init --recursive"
        )
    if "matcher" in sys.modules:
        raise RuntimeError(
            "package name collision: 'matcher' is already imported. "
            "Run GF-SAM and Matcher in separate processes "
            "(python -m src.methods.gfsam vs python -m src.methods.matcher)."
        )
    resolved = str(MATCHER_ROOT.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return MATCHER_ROOT


def matcher_oss_args(
    *,
    dinov2_weights: Path,
    sam_weights: Path,
    device: str,
) -> argparse.Namespace:
    """Paper / GETTING_STARTED OSS recipe — not ``main_oss.py`` defaults."""
    return argparse.Namespace(
        dinov2_size="vit_large",
        sam_size="vit_h",
        dinov2_weights=str(dinov2_weights),
        sam_weights=str(sam_weights),
        device=device,
        max_sample_iterations=64,
        box_nms_thresh=0.65,
        sample_range=(1, 6),
        topk_scores_threshold=0.0,
        use_dense_mask=1,
        use_points_or_centers=True,
        purity_filter=0.02,
        iou_filter=0.85,
        multimask_output=1,
        sel_stability_score_thresh=0.90,
        use_score_filter=True,
        alpha=1.0,
        beta=0.0,
        exp=0.0,
        num_merging_mask=9,
        points_per_side=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        output_layer=3,
        dense_multimask_output=0,
        num_centers=8,
        use_box=False,
        emd_filter=0.0,
        coverage_filter=0.0,
        deep_score_filter=0.33,
        deep_score_norm_filter=0.1,
    )


def build_matcher(
    *,
    dinov2_weights: Path | None = None,
    sam_weights: Path | None = None,
    image_size: int = 518,
    device: str = "cuda",
):
    add_matcher_to_syspath()
    dinov2_path = Path(dinov2_weights) if dinov2_weights is not None else DINOV2_VITL14_WEIGHTS
    sam_path = Path(sam_weights) if sam_weights is not None else SAM_VIT_H_WEIGHTS
    if not dinov2_path.is_file():
        raise SystemExit(f"DINOv2 weights missing: {dinov2_path}")
    if not sam_path.is_file():
        raise SystemExit(f"SAM ViT-H weights missing: {sam_path}")
    if image_size % 14 != 0:
        raise SystemExit(
            f"Matcher canvas must be divisible by 14 (DINOv2 patch size); got {image_size}. "
            "Use 518, not 768."
        )

    from matcher.Matcher import build_matcher_oss

    model = build_matcher_oss(matcher_oss_args(dinov2_weights=dinov2_path, sam_weights=sam_path, device=device))
    if image_size != 518:
        model.input_size = (image_size, image_size)
    return model
