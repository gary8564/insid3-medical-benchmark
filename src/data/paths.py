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

KVASIR_RAW = RAW_ROOT / "kvasir-seg"
KIPA_RAW = RAW_ROOT / "kipa22"
ACDC_RAW = RAW_ROOT / "acdc"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
