"""On-disk layout for raw archives and the processed 2D cache."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
PRETRAIN_ROOT = REPO_ROOT / "pretrain"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
INSID3_ROOT = THIRD_PARTY_ROOT / "INSID3"
GFSAM_ROOT = THIRD_PARTY_ROOT / "GF-SAM"
MATCHER_ROOT = THIRD_PARTY_ROOT / "Matcher"
FLEXICT_ROOT = THIRD_PARTY_ROOT / "FlexiCT"
MEDSAM2_ROOT = THIRD_PARTY_ROOT / "MedSAM2"

DINOV2_VITL14_WEIGHTS = PRETRAIN_ROOT / "dinov2_vitl14_pretrain.pth"
SAM_VIT_H_WEIGHTS = PRETRAIN_ROOT / "sam_vit_h_4b8939.pth"
FLEXICT_2D_WEIGHTS = PRETRAIN_ROOT / "flexict_2d_teacher.pth"
MEDSAM2_WEIGHTS = PRETRAIN_ROOT / "MedSAM2_latest.pt"

KVASIR_RAW = RAW_ROOT / "kvasir-seg"
KIPA_RAW = RAW_ROOT / "kipa22"
ACDC_RAW = RAW_ROOT / "acdc"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
