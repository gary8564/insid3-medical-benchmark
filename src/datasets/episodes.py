"""
Episodes from a processed 2D cache.

Following INSID3 `datasets/lung.py`, independently sample a random target and a different random reference from every paired image on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.constants import DEFAULT_SEED, DEFAULT_SHOTS, INSID3_N_EPISODES
from src.datasets.pairing import pair_images_and_masks

DOMAIN_DIRS = {
    "polyp": "polyp",
    "kidney_tumor": "kidney_tumor",
    "cardiac": "cardiac",
}


@dataclass(frozen=True)
class Episode:
    dataset: str
    reference_id: str
    target_id: str
    reference_image: Path
    reference_mask: Path
    target_image: Path
    target_mask: Path
    episode_index: int = 0


def processed_domain_root(data_root: Path | str, dataset: str) -> Path:
    if dataset not in DOMAIN_DIRS:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(DOMAIN_DIRS)}")
    return Path(data_root) / DOMAIN_DIRS[dataset]


def load_paired_index(data_root: Path | str, dataset: str) -> dict[str, tuple[Path, Path]]:
    root = processed_domain_root(data_root, dataset)
    pairs = pair_images_and_masks(root / "images", root / "masks")
    return {item_id: (image, mask) for item_id, image, mask in pairs}


def sample_insid3_id_pairs(
    ids: Sequence[str],
    *,
    n_episodes: int = INSID3_N_EPISODES,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, str]]:
    """
    Match INSID3 `DatasetLung.sample_episode` for 1-shot episodes (reference != target).

    `lung.py` draws with the global NumPy RNG after `np.random.seed(args.seed)`.
    `RandomState(seed)` is that same generator, isolated from other NumPy use.
    """
    if n_shots != 1:
        raise ValueError("only 1-shot sampling is implemented (INSID3 --shots default)")
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    pool = [str(x) for x in ids]
    if len(set(pool)) != len(pool):
        raise ValueError("ids must be unique")
    if len(pool) < 2:
        raise ValueError("need at least two images to sample a 1-shot episode")
    rng = np.random.RandomState(seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(n_episodes):
        target_id = str(rng.choice(pool, 1, replace=False)[0])
        while True:
            reference_id = str(rng.choice(pool, 1, replace=False)[0])
            if reference_id != target_id:
                pairs.append((reference_id, target_id))
                break
    return pairs


def sample_insid3_episodes(
    data_root: Path | str,
    dataset: str,
    *,
    n_episodes: int = INSID3_N_EPISODES,
    n_shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
) -> list[Episode]:
    """Random 1-shot episodes from every paired image on disk (INSID3 lung/ISIC)."""
    index = load_paired_index(data_root, dataset)
    pool = sorted(index)
    episodes: list[Episode] = []
    for episode_index, (reference_id, target_id) in enumerate(
        sample_insid3_id_pairs(pool, n_episodes=n_episodes, n_shots=n_shots, seed=seed)
    ):
        ref_image, ref_mask = index[reference_id]
        target_image, target_mask = index[target_id]
        episodes.append(
            Episode(
                dataset=dataset,
                reference_id=reference_id,
                target_id=target_id,
                reference_image=ref_image,
                reference_mask=ref_mask,
                target_image=target_image,
                target_mask=target_mask,
                episode_index=episode_index,
            )
        )
    return episodes


def select_preview_episodes(
    episodes: Sequence[Episode],
    n: int,
    seed: int = DEFAULT_SEED,
) -> list[Episode]:
    """Pick ``n`` episodes uniformly from the full sampled list (seeded).

    Indices come from a fresh ``RandomState(seed)``, not the pair-sampling
    stream, so the subset is a random draw from the headline 600 rather than
    its prefix. Selected episodes keep their original ``episode_index``.
    """
    if n < 1:
        raise ValueError("preview n must be >= 1")
    if n >= len(episodes):
        return list(episodes)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(episodes), size=n, replace=False)
    return [episodes[i] for i in sorted(chosen.tolist())]
