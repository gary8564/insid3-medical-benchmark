from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.episodes import sample_insid3_episodes
from src.protocol.loop import is_complete, run_episodes
from tests.conftest import write_domain_cache


def _fields(dataset: str = "polyp") -> dict[str, object]:
    return {
        "dataset": dataset,
        "protocol": "insid3_random",
        "seed": 0,
        "shots": 1,
        "method": "gfsam",
        "image_size": 32,
        "prompt": "1shot_mask",
        "tier": "B",
    }


def test_loop_writes_metrics_and_original_size_preds(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "t1"], size=(8, 8))
    episodes = sample_insid3_episodes(data_root, "polyp", n_episodes=2, seed=0)
    out = tmp_path / "results" / "gfsam" / "polyp"

    def predict(ep):
        return np.ones((32, 32), dtype=np.uint8)

    payload = run_episodes(episodes, predict, out_dir=out, metrics_fields=_fields())
    assert payload["n"] == 2
    assert (out / "metrics.json").is_file()
    assert (out / "run.json").is_file()
    assert (out / "episodes.json").is_file()
    pred = np.array(Image.open(next((out / "preds").glob("*.png"))))
    assert pred.shape == (8, 8)


def test_skip_if_complete(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "t1"], size=(8, 8))
    episodes = sample_insid3_episodes(data_root, "polyp", n_episodes=2, seed=0)
    out = tmp_path / "out"
    calls = {"n": 0}

    def predict(ep):
        calls["n"] += 1
        return np.zeros((8, 8), dtype=np.uint8)

    run_episodes(episodes, predict, out_dir=out, metrics_fields=_fields())
    assert calls["n"] == 2
    run_episodes(episodes, predict, out_dir=out, metrics_fields=_fields())
    assert calls["n"] == 2
    assert is_complete(out / "metrics.json", {"n": 2, "seed": 0, "image_size": 32, "method": "gfsam"})


def test_isolate_errors_writes_zero_mask(tmp_path: Path):
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "t1"], size=(8, 8))
    episodes = sample_insid3_episodes(data_root, "polyp", n_episodes=1, seed=0)
    out = tmp_path / "out"

    def predict(ep):
        raise RuntimeError("hungarian empty")

    payload = run_episodes(
        episodes, predict, out_dir=out, metrics_fields=_fields(), isolate_errors=True
    )
    assert payload["items"][0]["error"].startswith("RuntimeError")
    assert payload["items"][0]["IoU"] == 0.0
    row = json.loads((out / "metrics.json").read_text())
    assert "error" in row["items"][0]
