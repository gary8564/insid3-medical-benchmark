"""ACDC 2D cache: one ED LV-cavity slice per patient."""

from __future__ import annotations

from pathlib import Path

from src.datasets.episodes import load_paired_index


def list_pairs(data_root: Path | str) -> dict[str, tuple[Path, Path]]:
    return load_paired_index(data_root, "cardiac")
