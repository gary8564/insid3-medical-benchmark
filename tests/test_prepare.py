from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.constants import ACDC_LV_LABEL, KIPA_TUMOR_LABEL
from src.data.prepare import parse_args, run_prepare
from tests.conftest import write_gray_png, write_nifti


def _write_kvasir_raw(root: Path, ids: list[str]) -> Path:
    images = root / "Kvasir-SEG" / "images"
    masks = root / "Kvasir-SEG" / "masks"
    for item_id in ids:
        write_gray_png(images / f"{item_id}.png", np.full((8, 8), 70, dtype=np.uint8))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        write_gray_png(masks / f"{item_id}.png", mask)
    return root


def _write_kipa_raw(root: Path, ids: list[str]) -> Path:
    for i, case_id in enumerate(ids):
        image = np.zeros((8, 8, 4), dtype=np.float32)
        mask = np.zeros((8, 8, 4), dtype=np.uint8)
        image[:, :, 1] = 20.0 + i
        mask[0:3, 0:3, 1] = KIPA_TUMOR_LABEL
        write_nifti(root / "train" / "image" / f"{case_id}.nii.gz", image)
        write_nifti(root / "train" / "label" / f"{case_id}.nii.gz", mask)
    return root


def _write_acdc_raw(root: Path, names: list[str]) -> Path:
    for name in names:
        patient = root / "training" / name
        patient.mkdir(parents=True)
        (patient / "Info.cfg").write_text("ED: 1\nES: 8\n")
        image = np.zeros((8, 8, 3), dtype=np.float32)
        mask = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 1] = 12.0
        mask[1:4, 1:4, 1] = ACDC_LV_LABEL
        write_nifti(patient / f"{name}_frame01.nii.gz", image)
        write_nifti(patient / f"{name}_frame01_gt.nii.gz", mask)
    return root


def test_prepare_skip_download_writes_cache(tmp_path: Path):
    kvasir = _write_kvasir_raw(tmp_path / "kvasir", ["p0", "p1", "p2"])
    kipa = _write_kipa_raw(tmp_path / "kipa", ["0", "1"])
    acdc = _write_acdc_raw(tmp_path / "acdc", ["patient001", "patient002"])
    processed = tmp_path / "processed"

    args = parse_args(
        [
            "--skip-download",
            "--kvasir-raw",
            str(kvasir),
            "--kipa-raw",
            str(kipa),
            "--acdc-raw",
            str(acdc),
            "--processed-root",
            str(processed),
        ]
    )
    written = run_prepare(args)

    assert set(written) == {"polyp", "kidney_tumor", "cardiac"}
    assert written["polyp"] == ["p0", "p1", "p2"]
    assert Image.open(processed / "polyp" / "images" / "p0.png").size == (8, 8)
    assert (processed / "kidney_tumor" / "slice_stats.json").is_file()
    assert (processed / "cardiac" / "masks" / "patient001.png").is_file()
