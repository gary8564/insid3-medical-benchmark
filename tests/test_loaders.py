from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.cardiac_loader import list_pairs as list_cardiac
from src.datasets.episodes import (
    sample_insid3_episodes,
    sample_insid3_id_pairs,
    select_preview_episodes,
)
from src.datasets.kidney_tumor_loader import list_pairs as list_kidney
from src.datasets.pairing import pair_images_and_masks
from src.datasets.polyp_loader import list_pairs as list_polyp
from tests.conftest import write_domain_cache, write_gray_png


def test_polyp_pairs_images_and_masks_by_id(tmp_path: Path):
    ids = ["cju0u2x5v", "cju0u82sl", "cju0ue95v"]
    write_domain_cache(tmp_path, "polyp", ids)

    pairs = list_polyp(tmp_path)

    assert list(pairs) == sorted(ids)
    for item_id, (image, mask) in pairs.items():
        assert image.stem == mask.stem == item_id
        assert image.parent.name == "images"
        assert mask.parent.name == "masks"


def test_polyp_raises_when_a_mask_is_missing(tmp_path: Path):
    write_domain_cache(tmp_path, "polyp", ["keep", "orphan"])
    (tmp_path / "polyp" / "masks" / "orphan.png").unlink()

    with pytest.raises(FileNotFoundError, match="orphan"):
        pair_images_and_masks(tmp_path / "polyp" / "images", tmp_path / "polyp" / "masks")


def test_kidney_and_cardiac_use_the_same_pairing_rule(tmp_path: Path):
    write_domain_cache(tmp_path, "kidney_tumor", ["case_00000", "case_00001"])
    write_domain_cache(tmp_path, "cardiac", ["patient001", "patient002"])

    assert set(list_kidney(tmp_path)) == {"case_00000", "case_00001"}
    assert set(list_cardiac(tmp_path)) == {"patient001", "patient002"}


def test_insid3_id_pairs_match_lung_py_rng_stream():
    ids = ["a", "b", "c"]
    np.random.seed(0)
    expected: list[tuple[str, str]] = []
    for _ in range(12):
        target_id = str(np.random.choice(ids, 1, replace=False)[0])
        while True:
            reference_id = str(np.random.choice(ids, 1, replace=False)[0])
            if reference_id != target_id:
                expected.append((reference_id, target_id))
                break

    assert sample_insid3_id_pairs(ids, n_episodes=12, seed=0) == expected
    assert all(ref != tgt for ref, tgt in expected)


def test_insid3_episodes_sample_from_all_paired_images(tmp_path: Path):
    ids = ["ref", "t1", "t2"]
    write_domain_cache(tmp_path, "polyp", ids)

    built = sample_insid3_episodes(tmp_path, "polyp", n_episodes=40, seed=0)

    assert len(built) == 40
    assert all(ep.reference_id != ep.target_id for ep in built)
    assert {ep.reference_id for ep in built} <= set(ids)
    assert {ep.target_id for ep in built} <= set(ids)
    assert len({ep.reference_id for ep in built}) > 1
    assert [ep.episode_index for ep in built] == list(range(40))
    again = sample_insid3_episodes(tmp_path, "polyp", n_episodes=40, seed=0)
    assert [(a.reference_id, a.target_id) for a in again] == [
        (b.reference_id, b.target_id) for b in built
    ]


def test_select_preview_episodes_is_seeded_random_subset_not_prefix(tmp_path: Path):
    write_domain_cache(tmp_path, "polyp", ["a", "b", "c"])
    full = sample_insid3_episodes(tmp_path, "polyp", n_episodes=40, seed=0)
    preview = select_preview_episodes(full, 8, seed=0)
    again = select_preview_episodes(full, 8, seed=0)

    assert len(preview) == 8
    assert [ep.episode_index for ep in preview] == [ep.episode_index for ep in again]
    assert {ep.episode_index for ep in preview} <= set(range(40))
    assert [ep.episode_index for ep in preview] != list(range(8))
    by_index = {ep.episode_index: ep for ep in full}
    for ep in preview:
        match = by_index[ep.episode_index]
        assert (ep.reference_id, ep.target_id) == (match.reference_id, match.target_id)


def test_jpg_image_pairs_with_png_mask(tmp_path: Path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    write_gray_png(images / "a.png", np.zeros((4, 4), dtype=np.uint8))
    (images / "a.png").replace(images / "a.jpg")
    write_gray_png(masks / "a.png", np.zeros((4, 4), dtype=np.uint8))

    pairs = pair_images_and_masks(images, masks)
    assert pairs[0][0] == "a"
    assert pairs[0][1].suffix == ".jpg"
    assert pairs[0][2].suffix == ".png"
