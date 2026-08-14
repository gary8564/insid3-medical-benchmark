from __future__ import annotations

import numpy as np
import pytest

from PIL import Image

from src.data.constants import ACDC_LV_LABEL, KIPA_TUMOR_LABEL
from src.data.io_utils import infer_slice_axis, move_slice_axis_first, slice_to_uint8
from src.data.preprocess import (
    acdc_ed_frame_index,
    acdc_lv_slice,
    kipa_tumor_slice,
    parse_acdc_info,
    preprocess_acdc,
    preprocess_kipa,
    preprocess_kvasir,
)
from tests.conftest import write_gray_png, write_nifti


def test_kipa_selects_max_tumor_axial_slice_and_drops_other_labels():
    mask = np.zeros((6, 8, 8), dtype=np.uint8)
    mask[1, 0:2, 0:2] = KIPA_TUMOR_LABEL  # 4 tumor voxels
    mask[4, 0:4, 0:5] = KIPA_TUMOR_LABEL  # 20 tumor voxels (largest)
    mask[4, 6:8, 6:8] = 2  # kidney on the same slice; must not enter the binary mask
    mask[5] = 3  # artery only

    index, binary, area = kipa_tumor_slice(mask)

    assert index == 4
    assert area == 20
    assert binary.shape == (8, 8)
    assert set(np.unique(binary).tolist()) <= {0, 1}
    assert int(binary.sum()) == 20
    assert binary[6, 6] == 0  # kidney voxels excluded


def test_kipa_raises_when_tumor_absent():
    mask = np.zeros((3, 4, 4), dtype=np.uint8)
    mask[1] = 2
    with pytest.raises(ValueError, match="label 4"):
        kipa_tumor_slice(mask)


def test_acdc_selects_max_lv_slice_on_ed_volume():
    mask = np.zeros((5, 8, 8), dtype=np.uint8)
    mask[0, 0:2, 0:2] = ACDC_LV_LABEL  # 4
    mask[2, 0:4, 0:4] = ACDC_LV_LABEL  # 16 (largest)
    mask[2, 6:8, 0:2] = 1  # RV on the winning slice; must not enter LV mask
    mask[3] = 2  # myocardium only

    index, binary, area = acdc_lv_slice(mask)

    assert index == 2
    assert area == 16
    assert int(binary.sum()) == 16
    assert binary[6, 0] == 0  # RV excluded


def test_slice_to_uint8_percentile_stretch():
    plane = np.array([[0.0, 50.0], [100.0, 1000.0]], dtype=np.float32)
    out = slice_to_uint8(plane)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255
    assert out[0, 0] < out[1, 1]


def test_move_slice_axis_first_puts_smallest_axis_on_axis_0():
    volume = np.zeros((16, 16, 5), dtype=np.uint8)
    assert infer_slice_axis(volume, plane="acquisition") == 2
    stacked = move_slice_axis_first(volume)
    assert stacked.shape == (5, 16, 16)


def test_infer_slice_axis_anatomical_planes_follow_ras_affine():
    volume = np.zeros((32, 32, 40), dtype=np.uint8)
    ras = np.eye(4)
    assert infer_slice_axis(volume, ras, plane="sagittal") == 0
    assert infer_slice_axis(volume, ras, plane="coronal") == 1
    assert infer_slice_axis(volume, ras, plane="axial") == 2

    # Voxel axis 0 is superior–inferior (S, A, R).
    s_on_axis0 = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert infer_slice_axis(volume, s_on_axis0, plane="axial") == 0


def test_infer_slice_axis_acquisition_uses_thickest_spacing_not_smallest_shape():
    volume = np.zeros((32, 32, 40), dtype=np.uint8)
    thick_last = np.diag([1.0, 1.0, 8.0, 1.0])
    assert infer_slice_axis(volume, thick_last, plane="acquisition") == 2
    isotropic = np.eye(4)
    assert infer_slice_axis(volume, isotropic, plane="acquisition") == 0


def test_infer_slice_axis_rejects_unknown_plane():
    with pytest.raises(ValueError, match="plane"):
        infer_slice_axis(np.zeros((4, 4, 4)), plane="stack")


def test_parse_acdc_ed_frame(tmp_path):
    path = tmp_path / "Info.cfg"
    path.write_text("ED: 1\nES: 12\nNbFrame: 30\n")
    info = parse_acdc_info(path)
    assert info["ES"] == "12"
    assert acdc_ed_frame_index(info) == 1


def test_acdc_ed_frame_index_rejects_missing_and_non_numeric():
    with pytest.raises(KeyError, match="end-diastole"):
        acdc_ed_frame_index({"ES": "12"})
    with pytest.raises(ValueError, match="integer frame number"):
        acdc_ed_frame_index({"ED": "late"})


def test_preprocess_kvasir_copies_images_and_binarizes_masks(tmp_path):
    raw = tmp_path / "raw" / "Kvasir-SEG"
    images = raw / "images"
    masks = raw / "masks"
    write_gray_png(images / "a.png", np.full((8, 8), 80, dtype=np.uint8))
    write_gray_png(images / "b.png", np.full((8, 8), 90, dtype=np.uint8))
    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_a[2:5, 2:5] = 200
    write_gray_png(masks / "a.png", mask_a)
    write_gray_png(masks / "b.png", np.zeros((8, 8), dtype=np.uint8))

    ids = preprocess_kvasir(tmp_path / "raw", tmp_path / "processed")
    assert ids == ["a", "b"]
    out_mask = np.array(Image.open(tmp_path / "processed" / "polyp" / "masks" / "a.png"))
    assert set(np.unique(out_mask).tolist()) <= {0, 255}
    assert int((out_mask > 0).sum()) == 9


def test_preprocess_kipa_writes_max_tumor_png_and_stats(tmp_path):
    raw = tmp_path / "raw" / "train"
    for case_id, tumor_slice in (("0", 1), ("1", 2)):
        image = np.zeros((8, 8, 4), dtype=np.float32)
        image[:, :, tumor_slice] = 40.0
        mask = np.zeros((8, 8, 4), dtype=np.uint8)
        mask[0:3, 0:3, tumor_slice] = KIPA_TUMOR_LABEL
        mask[6:8, 6:8, tumor_slice] = 2
        write_nifti(raw / "image" / f"{case_id}.nii.gz", image)
        write_nifti(raw / "label" / f"{case_id}.nii.gz", mask)

    ids, stats = preprocess_kipa(tmp_path / "raw", tmp_path / "processed")
    assert ids == ["0", "1"]
    assert [row["slice_index"] for row in stats] == [1, 2]
    assert stats[0]["tumor_pixels"] == 9
    mask = np.array(Image.open(tmp_path / "processed" / "kidney_tumor" / "masks" / "0.png"))
    assert mask[6, 6] == 0
    assert (tmp_path / "processed" / "kidney_tumor" / "slice_stats.json").is_file()


def test_preprocess_acdc_uses_ed_frame_and_lv_label(tmp_path):
    training = tmp_path / "raw" / "training"
    for name, ed in (("patient001", 1), ("patient002", 2)):
        patient = training / name
        patient.mkdir(parents=True)
        (patient / "Info.cfg").write_text(f"ED: {ed}\nES: 9\n")
        image = np.zeros((8, 8, 4), dtype=np.float32)
        image[:, :, 2] = 30.0
        mask = np.zeros((8, 8, 4), dtype=np.uint8)
        mask[1:5, 1:5, 2] = ACDC_LV_LABEL
        mask[6:8, 0:2, 2] = 1
        write_nifti(patient / f"{name}_frame{ed:02d}.nii.gz", image)
        write_nifti(patient / f"{name}_frame{ed:02d}_gt.nii.gz", mask)

    ids = preprocess_acdc(tmp_path / "raw", tmp_path / "processed")
    assert ids == ["patient001", "patient002"]
    lv = np.array(Image.open(tmp_path / "processed" / "cardiac" / "masks" / "patient001.png"))
    assert int((lv > 0).sum()) == 16
    assert lv[6, 0] == 0

