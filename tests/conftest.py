"""Shared helpers for writing tiny PNG caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_gray_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(path)


def write_nifti(path: Path, array: np.ndarray, affine: np.ndarray | None = None) -> Path:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine), str(path))
    return path


def write_domain_cache(
    root: Path,
    dataset: str,
    ids: list[str],
    size: tuple[int, int] = (8, 8),
) -> Path:
    """Write matching image/mask PNGs under ``root / dataset / {images,masks}``."""
    height, width = size
    image_dir = root / dataset / "images"
    mask_dir = root / dataset / "masks"
    for i, item_id in enumerate(ids):
        image = np.full((height, width), 40 + 10 * i, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        write_gray_png(image_dir / f"{item_id}.png", image)
        write_gray_png(mask_dir / f"{item_id}.png", mask)
    return root / dataset
