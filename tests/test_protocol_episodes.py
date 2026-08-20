from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.episodes import sample_insid3_episodes
from src.protocol.episodes_io import load_persisted_episodes, serialize_episodes, write_json
from tests.conftest import write_domain_cache


def test_roundtrip_phase3_list(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "t1", "t2"])
    episodes = sample_insid3_episodes(data_root, "polyp", n_episodes=3, seed=0)
    path = tmp_path / "episodes.json"
    write_json(path, serialize_episodes(episodes))

    loaded = load_persisted_episodes(path, input_dir=data_root, dataset="polyp")
    assert len(loaded) == 3
    assert [ep.target_id for ep in loaded] == [ep.target_id for ep in episodes]
    assert [ep.reference_ids for ep in loaded] == [ep.reference_ids for ep in episodes]
    assert loaded[0].target_image.is_file()


def test_remap_when_stored_paths_are_gone(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["a", "b", "c"])
    episodes = sample_insid3_episodes(data_root, "polyp", n_episodes=2, seed=0)
    rows = serialize_episodes(episodes)
    for row in rows:
        row["reference_images"] = [p.replace("processed", "missing") for p in row["reference_images"]]
        row["reference_masks"] = [p.replace("processed", "missing") for p in row["reference_masks"]]
        row["target_image"] = row["target_image"].replace("processed", "missing")
        row["target_mask"] = row["target_mask"].replace("processed", "missing")
    path = tmp_path / "episodes.json"
    write_json(path, rows)

    loaded = load_persisted_episodes(path, input_dir=data_root)
    assert loaded[0].target_image.is_file()
    assert loaded[0].target_id == episodes[0].target_id


def test_missing_id_fails_loudly(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["a", "b"])
    path = tmp_path / "episodes.json"
    write_json(
        path,
        [
            {
                "episode_index": 0,
                "dataset": "polyp",
                "reference_ids": ["missing_ref"],
                "target_id": "a",
                "reference_images": [str(tmp_path / "nope.png")],
                "reference_masks": [str(tmp_path / "nope.png")],
                "target_image": str(tmp_path / "nope.png"),
                "target_mask": str(tmp_path / "nope.png"),
            }
        ],
    )
    with pytest.raises(KeyError, match="missing_ref"):
        load_persisted_episodes(path, input_dir=data_root)


def test_legacy_scalar_reference_id(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "tgt"])
    image = data_root / "polyp" / "images" / "ref.png"
    mask = data_root / "polyp" / "masks" / "ref.png"
    tgt_image = data_root / "polyp" / "images" / "tgt.png"
    tgt_mask = data_root / "polyp" / "masks" / "tgt.png"
    path = tmp_path / "episodes.json"
    write_json(
        path,
        [
            {
                "episode_index": 7,
                "dataset": "polyp",
                "reference_id": "ref",
                "target_id": "tgt",
                "reference_image": str(image),
                "reference_mask": str(mask),
                "target_image": str(tgt_image),
                "target_mask": str(tgt_mask),
            }
        ],
    )
    loaded = load_persisted_episodes(path, input_dir=data_root)
    assert loaded[0].reference_ids == ("ref",)
    assert loaded[0].episode_index == 7
