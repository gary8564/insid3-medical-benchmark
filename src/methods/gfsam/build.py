"""Put ``third_party/GF-SAM`` on ``sys.path`` and build a ``GFSAM`` instance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.paths import DINOV2_VITL14_WEIGHTS, GFSAM_ROOT, SAM_VIT_H_WEIGHTS


def add_gfsam_to_syspath() -> Path:
    if not (GFSAM_ROOT / "matcher" / "GFSAM.py").is_file():
        raise FileNotFoundError(
            f"GF-SAM submodule is missing at {GFSAM_ROOT}. "
            "From the repo root run: git submodule update --init --recursive"
        )
    if "matcher" in sys.modules:
        raise RuntimeError(
            "package name collision: 'matcher' is already imported. "
            "Run GF-SAM and Matcher in separate processes "
            "(python -m src.methods.gfsam vs python -m src.methods.matcher)."
        )
    resolved = str(GFSAM_ROOT.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return GFSAM_ROOT


def build_gfsam(
    *,
    dinov2_weights: Path | None = None,
    sam_weights: Path | None = None,
    image_size: int = 1024,
    device: str = "cuda",
):
    add_gfsam_to_syspath()
    dinov2_path = Path(dinov2_weights) if dinov2_weights is not None else DINOV2_VITL14_WEIGHTS
    sam_path = Path(sam_weights) if sam_weights is not None else SAM_VIT_H_WEIGHTS
    if not dinov2_path.is_file():
        raise SystemExit(f"DINOv2 weights missing: {dinov2_path}")
    if not sam_path.is_file():
        raise SystemExit(f"SAM ViT-H weights missing: {sam_path}")

    from matcher.GFSAM import build_model

    args = argparse.Namespace(
        dinov2_size="vit_large",
        sam_size="vit_h",
        dinov2_weights=str(dinov2_path),
        sam_weights=str(sam_path),
        device=device,
    )
    if image_size == 1024:
        return build_model(args)
    # build_model does not pass input_size; GFSAM defaults to 1024.
    model = build_model(args)
    model.input_size = (image_size, image_size)
    return model
