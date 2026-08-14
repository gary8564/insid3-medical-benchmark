"""Match image stems to mask stems in a processed 2D folder."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff"}


def _index_by_stem(directory: Path, allowed: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in allowed or not path.is_file():
            continue
        if path.stem in indexed:
            raise ValueError(f"duplicate stem {path.stem!r} in {directory}")
        indexed[path.stem] = path
    return indexed


def pair_images_and_masks(
    image_dir: Path | str,
    mask_dir: Path | str,
) -> list[tuple[str, Path, Path]]:
    """Return (id, image_path, mask_path) for stems present in both folders."""
    images = _index_by_stem(Path(image_dir), IMAGE_EXTENSIONS)
    masks = _index_by_stem(Path(mask_dir), MASK_EXTENSIONS)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks:
        raise FileNotFoundError(f"images without masks: {missing_masks}")
    if missing_images:
        raise FileNotFoundError(f"masks without images: {missing_images}")
    return [(stem, images[stem], masks[stem]) for stem in sorted(images)]
