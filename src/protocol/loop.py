"""Shared episode loop: skip-if-complete, predict, write metrics / preds / run.json."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data.io_utils import save_binary_mask_png
from src.datasets.episodes import Episode
from src.evaluate import binary_dice, binary_iou, mean_metrics
from src.protocol.episodes_io import serialize_episodes, write_json
from src.protocol.masks import as_binary_mask, load_binary_mask, resize_pred_to_gt
from src.protocol.run_meta import collect_run_meta

PredictFn = Callable[[Episode], np.ndarray]


def is_complete(metrics_path: Path, expected: dict[str, object]) -> bool:
    if not metrics_path.is_file():
        return False
    row = json.loads(metrics_path.read_text())
    if int(row.get("n", 0)) != int(expected["n"]):
        return False
    for key in ("seed", "image_size"):
        if key in expected and int(row.get(key, -1)) != int(expected[key]):
            return False
    for key in ("method", "model", "model_size", "backbone", "debiased", "prompt"):
        if key in expected and key in row and row[key] != expected[key]:
            return False
    return True


def run_episodes(
    episodes: list[Episode],
    predict_fn: PredictFn,
    *,
    out_dir: Path,
    metrics_fields: dict[str, object],
    overwrite: bool = False,
    isolate_errors: bool = False,
) -> dict[str, object] | None:
    """Run ``predict_fn`` on every episode and write ``metrics.json``.

    ``predict_fn`` returns a 2D mask on any canvas; this loop nearest-resizes
    it to the original GT PNG before scoring. Prints only dataset / n / mIoU / Dice.
    """
    if not episodes:
        raise ValueError("no episodes to run")

    out_dir = Path(out_dir)
    pred_dir = out_dir / "preds"
    metrics_path = out_dir / "metrics.json"
    expected = {
        "n": len(episodes),
        **{key: metrics_fields[key] for key in ("seed", "image_size", "method", "model_size", "backbone", "debiased", "prompt") if key in metrics_fields},
    }
    if not overwrite and is_complete(metrics_path, expected):
        row = json.loads(metrics_path.read_text())
        sys.stdout.write(
            f"{row.get('dataset', metrics_fields.get('dataset'))}: skip "
            f"n={row['n']}  mIoU={row['mIoU']:.3f}  Dice={row['Dice']:.3f}  → {metrics_path}\n"
        )
        return row

    pred_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "episodes.json", serialize_episodes(episodes))
    write_json(out_dir / "run.json", collect_run_meta())

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    per_item: list[dict[str, object]] = []
    iterator = episodes
    if len(episodes) > 1:
        from tqdm import tqdm

        iterator = tqdm(episodes, desc=str(metrics_fields.get("dataset", "eval")))

    for ep in iterator:
        gt = load_binary_mask(ep.target_mask)
        error = None
        try:
            pred = resize_pred_to_gt(as_binary_mask(predict_fn(ep)), gt.shape)
        except Exception as exc:
            if not isolate_errors:
                raise
            error = f"{type(exc).__name__}: {exc}"
            pred = np.zeros(gt.shape, dtype=np.uint8)
        pred_name = f"{ep.episode_index:04d}_{ep.target_id}.png"
        save_binary_mask_png(pred_dir / pred_name, pred)
        iou = binary_iou(pred, gt)
        dice = binary_dice(pred, gt)
        if not math.isfinite(iou) or not math.isfinite(dice):
            raise ValueError(f"non-finite metric for {ep.target_id}: IoU={iou} Dice={dice}")
        pairs.append((pred, gt))
        item: dict[str, object] = {
            "episode_index": ep.episode_index,
            "reference_ids": list(ep.reference_ids),
            "target_id": ep.target_id,
            "IoU": iou,
            "Dice": dice,
        }
        if error is not None:
            item["error"] = error
        per_item.append(item)

    summary = mean_metrics(pairs)
    payload: dict[str, object] = {
        "dataset": metrics_fields.get("dataset", episodes[0].dataset),
        "protocol": metrics_fields.get("protocol", "insid3_random"),
        "seed": metrics_fields.get("seed"),
        "shots": metrics_fields.get("shots", len(episodes[0].reference_ids)),
        "n": int(summary["n"]),
        "mIoU": summary["mIoU"],
        "Dice": summary["Dice"],
        "items": per_item,
        "pred_dir": str(pred_dir),
    }
    for key, value in metrics_fields.items():
        if key not in payload:
            payload[key] = value
    write_json(metrics_path, payload)
    sys.stdout.write(
        f"{payload['dataset']}: n={payload['n']}  "
        f"mIoU={payload['mIoU']:.3f}  Dice={payload['Dice']:.3f}  → {metrics_path}\n"
    )
    return payload
