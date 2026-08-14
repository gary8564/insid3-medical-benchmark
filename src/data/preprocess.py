"""
Slice rules for the processed 2D cache.
3D volumes are indexed as (D, H, W) (depth, height, width).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.constants import ACDC_LV_LABEL, KIPA_TUMOR_LABEL
from src.data.io_utils import (
    find_named_dir,
    load_nifti,
    slice_axis_for_pair,
    save_binary_mask_png,
    save_gray_png,
    slice_to_uint8,
    move_slice_axis_first,
)
from src.datasets.pairing import pair_images_and_masks


def _as_3d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got shape {mask.shape}")
    return np.asarray(mask)


def slice_areas(mask: np.ndarray, label: int, slice_axis: int = 0) -> np.ndarray:
    """Foreground voxel count equal to label on each slice along slice_axis."""
    volume = _as_3d(mask)
    binary = volume == label
    reduce_axes = tuple(i for i in range(volume.ndim) if i != slice_axis)
    return binary.sum(axis=reduce_axes)


def select_max_area_slice_index(mask: np.ndarray, label: int, slice_axis: int = 0) -> int:
    """Select the slice with the maximum foreground voxel count of the given label."""
    areas = slice_areas(mask, label, slice_axis=slice_axis)
    if int(areas.max()) == 0:
        raise ValueError(f"label {label} has no voxels in the volume")
    return int(np.argmax(areas))


def extract_binary_slice(
    mask: np.ndarray,
    slice_index: int,
    label: int,
    slice_axis: int = 0,
) -> np.ndarray:
    """Extract the binary mask of the given label on the specified slice."""
    plane = np.take(_as_3d(mask), slice_index, axis=slice_axis)
    return (plane == label).astype(np.uint8)


def kipa_tumor_slice(
    mask: np.ndarray, slice_axis: int = 0
) -> tuple[int, np.ndarray, int]:
    """Return (slice_index, binary tumor mask, tumor pixel area)."""
    index = select_max_area_slice_index(mask, KIPA_TUMOR_LABEL, slice_axis=slice_axis)
    binary = extract_binary_slice(mask, index, KIPA_TUMOR_LABEL, slice_axis=slice_axis)
    return index, binary, int(binary.sum())


def acdc_lv_slice(mask: np.ndarray, slice_axis: int = 0) -> tuple[int, np.ndarray, int]:
    """Return (slice_index, binary LV-cavity mask, LV pixel area) on an ED volume."""
    index = select_max_area_slice_index(mask, ACDC_LV_LABEL, slice_axis=slice_axis)
    binary = extract_binary_slice(mask, index, ACDC_LV_LABEL, slice_axis=slice_axis)
    return index, binary, int(binary.sum())


def parse_acdc_info(path: Path | str) -> dict[str, str]:
    """
    Parse an ACDC patient Info.cfg.
    ACDC cine has two annotated cardiac phases:
    - ED (end-diastole): heart at maximum fill; largest LV cavity. 
    - ES (end-systole): peak contraction; smallest LV cavity. 
    """
    info: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def acdc_ed_frame_index(info: dict[str, str]) -> int:
    """Return the end-diastole cine frame number from parsed Info.cfg."""
    if "ED" not in info:
        raise KeyError("Info.cfg is missing ED (end-diastole frame number)")
    raw = info["ED"].strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Info.cfg ED must be an integer frame number, got {raw!r}"
        ) from exc


def preprocess_kvasir(raw_root: Path, processed_root: Path) -> list[str]:
    """
    Write one 2D polyp pair per Kvasir image (already 2D; no slice picking).
    Images are copied as-is. Masks are saved as binary PNGs (polyp vs background).
    """
    image_dir = find_named_dir(raw_root, "images")
    mask_dir = find_named_dir(raw_root, "masks")
    pairs = pair_images_and_masks(image_dir, mask_dir)
    out_images = processed_root / "polyp" / "images"
    out_masks = processed_root / "polyp" / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for item_id, image_path, mask_path in pairs:
        (out_images / f"{item_id}{image_path.suffix.lower()}").write_bytes(image_path.read_bytes())
        mask = np.array(Image.open(mask_path).convert("L"))
        save_binary_mask_png(out_masks / f"{item_id}.png", mask)
        ids.append(item_id)
    return ids


def preprocess_kipa(raw_root: Path, processed_root: Path) -> tuple[list[str], list[dict[str, int]]]:
    """
    Write one axial slice with the most tumor pixels per KiPA case.
    Other labels (vein, kidney, artery) become background. Tumor pixel area is recorded in kidney_tumor/slice_stats.json.
    """
    image_dir = find_named_dir(raw_root, "image")
    label_dir = find_named_dir(raw_root, "label")
    out_images = processed_root / "kidney_tumor" / "images"
    out_masks = processed_root / "kidney_tumor" / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    stats: list[dict[str, int]] = []
    ids: list[str] = []
    for label_path in sorted(label_dir.glob("*.nii.gz")):
        case_id = label_path.name.replace(".nii.gz", "")
        image_path = image_dir / label_path.name
        if not image_path.is_file():
            raise FileNotFoundError(f"KiPA image missing for {case_id}: {image_path}")
        mask_vol, mask_affine = load_nifti(label_path)
        image_vol, image_affine = load_nifti(image_path)
        axis = slice_axis_for_pair(
            mask_vol, mask_affine, image_vol, image_affine, plane="axial", label=case_id
        )
        mask = move_slice_axis_first(mask_vol, slice_axis=axis)
        image = move_slice_axis_first(image_vol, slice_axis=axis)
        if image.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{case_id}: image stack {image.shape} vs mask stack {mask.shape}"
            )
        index, binary, area = kipa_tumor_slice(mask, slice_axis=0)
        save_gray_png(out_images / f"{case_id}.png", slice_to_uint8(image[index]))
        save_binary_mask_png(out_masks / f"{case_id}.png", binary)
        ids.append(case_id)
        stats.append({"id": case_id, "slice_index": index, "tumor_pixels": area})
    stats_path = processed_root / "kidney_tumor" / "slice_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return ids, stats


def preprocess_acdc(raw_root: Path, processed_root: Path) -> list[str]:
    """
    Write one slice with the largest LV cavity per ACDC training patient.
    Info.cfg selects the end-diastole cine frame; then the slice with the largest LV cavity (label 3) is kept. RV and myocardium become background.
    """
    training = raw_root / "training"
    if not training.is_dir():
        training = find_named_dir(raw_root, "training")
    out_images = processed_root / "cardiac" / "images"
    out_masks = processed_root / "cardiac" / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for patient_dir in sorted(p for p in training.iterdir() if p.is_dir() and p.name.startswith("patient")):
        info = parse_acdc_info(patient_dir / "Info.cfg")
        ed = acdc_ed_frame_index(info)
        stem = f"{patient_dir.name}_frame{ed:02d}"
        image_path = patient_dir / f"{stem}.nii.gz"
        mask_path = patient_dir / f"{stem}_gt.nii.gz"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing ED frame for {patient_dir.name}: {image_path.name}")
        mask_vol, mask_affine = load_nifti(mask_path)
        image_vol, image_affine = load_nifti(image_path)
        axis = slice_axis_for_pair(
            mask_vol,
            mask_affine,
            image_vol,
            image_affine,
            plane="acquisition",
            label=patient_dir.name,
        )
        mask = move_slice_axis_first(mask_vol, slice_axis=axis)
        image = move_slice_axis_first(image_vol, slice_axis=axis)
        if image.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{patient_dir.name}: image stack {image.shape} vs mask stack {mask.shape}"
            )
        index, binary, _area = acdc_lv_slice(mask, slice_axis=0)
        save_gray_png(out_images / f"{patient_dir.name}.png", slice_to_uint8(image[index]))
        save_binary_mask_png(out_masks / f"{patient_dir.name}.png", binary)
        ids.append(patient_dir.name)
    return ids
