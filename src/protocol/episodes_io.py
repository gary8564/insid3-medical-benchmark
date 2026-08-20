"""Load Phase 3 ``episodes.json`` and remap paths if the Drive mount moved."""

from __future__ import annotations

import json
from pathlib import Path

from src.datasets.episodes import Episode, load_paired_index, select_preview_episodes


def serialize_episodes(episodes: list[Episode]) -> list[dict[str, object]]:
    return [
        {
            "episode_index": ep.episode_index,
            "dataset": ep.dataset,
            "reference_ids": list(ep.reference_ids),
            "target_id": ep.target_id,
            "reference_images": [str(path) for path in ep.reference_images],
            "reference_masks": [str(path) for path in ep.reference_masks],
            "target_image": str(ep.target_image),
            "target_mask": str(ep.target_mask),
        }
        for ep in episodes
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _as_id_list(row: dict) -> list[str]:
    if "reference_ids" in row:
        ids = row["reference_ids"]
        if isinstance(ids, str):
            return [ids]
        return [str(item) for item in ids]
    if "reference_id" in row:
        return [str(row["reference_id"])]
    raise KeyError("episode row is missing reference_ids / reference_id")


def _as_path_list(row: dict, key_plural: str, key_singular: str) -> list[Path]:
    if key_plural in row:
        values = row[key_plural]
        if isinstance(values, str):
            return [Path(values)]
        return [Path(item) for item in values]
    if key_singular in row:
        return [Path(row[key_singular])]
    return []


def _resolve_pair(
    item_id: str,
    stored_image: Path | None,
    stored_mask: Path | None,
    index: dict[str, tuple[Path, Path]] | None,
) -> tuple[Path, Path]:
    if stored_image is not None and stored_mask is not None:
        if stored_image.is_file() and stored_mask.is_file():
            return stored_image, stored_mask
    if index is None:
        raise FileNotFoundError(
            f"stored paths for {item_id!r} are missing and no --input-dir was given to remap them"
        )
    if item_id not in index:
        raise KeyError(
            f"episode id {item_id!r} is not in the processed cache; "
            "do not drop episodes — fix the path or the id"
        )
    return index[item_id]


def load_persisted_episodes(
    path: Path | str,
    input_dir: Path | str | None = None,
    dataset: str | None = None,
) -> list[Episode]:
    """Reload Phase 3 episodes. Never re-sample.

    Path order: use the stored file if it still exists, else rebuild from
    ``input_dir/<dataset>/{images,masks}`` via ``load_paired_index`` and the
    stored ids. Missing ids raise; episodes are never dropped.
    """
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        rows = payload.get("episodes")
        if rows is None:
            raise ValueError(f"{path}: expected a list or an object with an 'episodes' key")
        dataset = dataset or payload.get("dataset")
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: episodes list is empty")

    dataset = dataset or str(rows[0].get("dataset") or "")
    if not dataset:
        raise ValueError(f"{path}: dataset is missing from the JSON and was not passed in")

    index = None
    if input_dir is not None:
        index = load_paired_index(input_dir, dataset)

    episodes: list[Episode] = []
    for row in rows:
        reference_ids = tuple(_as_id_list(row))
        target_id = str(row["target_id"])
        stored_images = _as_path_list(row, "reference_images", "reference_image")
        stored_masks = _as_path_list(row, "reference_masks", "reference_mask")
        if stored_images and len(stored_images) != len(reference_ids):
            stored_images = []
        if stored_masks and len(stored_masks) != len(reference_ids):
            stored_masks = []

        ref_images: list[Path] = []
        ref_masks: list[Path] = []
        for i, ref_id in enumerate(reference_ids):
            image, mask = _resolve_pair(
                ref_id,
                stored_images[i] if i < len(stored_images) else None,
                stored_masks[i] if i < len(stored_masks) else None,
                index,
            )
            ref_images.append(image)
            ref_masks.append(mask)

        stored_tgt_image = Path(row["target_image"]) if row.get("target_image") else None
        stored_tgt_mask = Path(row["target_mask"]) if row.get("target_mask") else None
        target_image, target_mask = _resolve_pair(
            target_id, stored_tgt_image, stored_tgt_mask, index
        )
        episodes.append(
            Episode(
                dataset=dataset,
                reference_ids=reference_ids,
                target_id=target_id,
                reference_images=tuple(ref_images),
                reference_masks=tuple(ref_masks),
                target_image=target_image,
                target_mask=target_mask,
                episode_index=int(row.get("episode_index", len(episodes))),
            )
        )
    return episodes


def maybe_preview(episodes: list[Episode], preview: int | None, seed: int) -> list[Episode]:
    if preview is None:
        return episodes
    if preview < 1:
        raise ValueError("--preview must be >= 1")
    return select_preview_episodes(episodes, preview, seed=seed)
