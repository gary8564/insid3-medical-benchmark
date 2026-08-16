from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.backbones.dinov3_backbone import can_run_insid3, clamp_svd_components
from src.run_insid3 import main
from tests.conftest import write_domain_cache


def test_clamp_svd_components_fits_vit_s_and_patch_grid():
    assert clamp_svd_components("small", 256, 500) == 256
    assert clamp_svd_components("small", 256, 64) == 64
    assert clamp_svd_components("large", 768, 500) == 500


def test_insid3_one_pair(tmp_path: Path):
    """One synthetic pair through real INSID3 + local ViT-S (CPU)."""
    pytest.importorskip("torch")
    if not can_run_insid3("small"):
        pytest.skip("need third_party/INSID3 and pretrain/ ViT-S weights")

    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ["ref", "t1"], size=(32, 32))
    assert (
        main(
            [
                "--dataset",
                "polyp",
                "--input-dir",
                str(data_root),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-size",
                "small",
                "--image-size",
                "256",
                "--svd-comps",
                "64",
                "--preview",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    pred = np.array(Image.open(next((tmp_path / "out" / "preds").glob("*.png"))))
    assert pred.shape == (32, 32)
    assert set(np.unique((pred > 0).astype(np.uint8)).tolist()) <= {0, 1}
    assert math.isfinite(metrics["mIoU"])
    assert metrics["n"] == 1
