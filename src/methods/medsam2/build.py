"""Import MedSAM2 via ``sys.path`` only. Never compile ``sam2._C``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.data.paths import MEDSAM2_ROOT, MEDSAM2_WEIGHTS

DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"


def add_medsam2_to_syspath() -> Path:
    if not (MEDSAM2_ROOT / "sam2").is_dir():
        raise FileNotFoundError(
            f"MedSAM2 submodule is missing at {MEDSAM2_ROOT}. "
            "From the repo root run: git submodule update --init --recursive"
        )
    resolved = str(MEDSAM2_ROOT.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return MEDSAM2_ROOT


def build_medsam2_predictor(
    *,
    checkpoint: Path | None = None,
    cfg: str = DEFAULT_CFG,
):
    """``build_sam2_video_predictor_npz`` with Hydra configs resolved from the vendor tree."""
    root = add_medsam2_to_syspath()
    ckpt = Path(checkpoint) if checkpoint is not None else MEDSAM2_WEIGHTS
    if not ckpt.is_file():
        raise SystemExit(
            f"MedSAM2 checkpoint missing: {ckpt}. "
            "Download MedSAM2_latest.pt (see their download.sh) into pretrain/."
        )
    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    cwd = Path.cwd()
    try:
        os.chdir(root)
        from sam2.build_sam import build_sam2_video_predictor_npz

        return build_sam2_video_predictor_npz(cfg, str(ckpt))
    finally:
        os.chdir(cwd)
