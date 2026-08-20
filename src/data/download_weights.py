"""Download public Phase 6–8 checkpoints into pretrain/ (skip files that already exist)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.io_utils import download_url
from src.data.paths import (
    DINOV2_VITL14_WEIGHTS,
    FLEXICT_2D_WEIGHTS,
    MEDSAM2_WEIGHTS,
    PRETRAIN_ROOT,
    SAM_VIT_H_WEIGHTS,
    ensure_dir,
)

DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
SAM_VIT_H_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MEDSAM2_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
FLEXICT_DRIVE = "https://drive.google.com/file/d/1nUj2RCsNQfOAncMYY5S-YgQthteoAdSM/view?usp=drive_link"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=PRETRAIN_ROOT)
    parser.add_argument("--dinov2", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--medsam2", action="store_true")
    return parser.parse_args(argv)


def _dest_file(dest: Path, default: Path) -> Path:
    return dest / default.name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest = ensure_dir(args.dest)
    want_dinov2 = args.dinov2
    want_sam = args.sam
    want_medsam2 = args.medsam2
    if not (want_dinov2 or want_sam or want_medsam2):
        sys.stderr.write(
            "Nothing selected. Use --dinov2, --sam, --medsam2, or --all-public.\n"
            f"FlexiCT-2D is not public HTTP: download {FLEXICT_DRIVE}\n"
            f"and save it as {FLEXICT_2D_WEIGHTS.name} under {dest}.\n"
        )
        return 1
    if want_dinov2:
        path = download_url(DINOV2_URL, _dest_file(dest, DINOV2_VITL14_WEIGHTS))
        print(f"dinov2: {path} ({path.stat().st_size} bytes)")
    if want_sam:
        path = download_url(SAM_VIT_H_URL, _dest_file(dest, SAM_VIT_H_WEIGHTS))
        print(f"sam: {path} ({path.stat().st_size} bytes)")
    if want_medsam2:
        path = download_url(MEDSAM2_URL, _dest_file(dest, MEDSAM2_WEIGHTS))
        print(f"medsam2: {path} ({path.stat().st_size} bytes)")
    flexict = dest / FLEXICT_2D_WEIGHTS.name
    if not flexict.is_file():
        print(
            f"FlexiCT-2D (optional, Phase 7): save {FLEXICT_2D_WEIGHTS.name} to {dest}\n"
            f"  {FLEXICT_DRIVE}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
