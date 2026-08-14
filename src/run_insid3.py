"""INSID3 eval entry point. `--dry-run` builds episodes without loading DINOv3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.constants import (
    DEFAULT_SEED,
    DEFAULT_SHOTS,
    INSID3_N_EPISODES,
    PREVIEW_N,
)
from src.data.io_utils import save_binary_mask_png
from src.datasets.episodes import (
    DOMAIN_DIRS,
    Episode,
    sample_insid3_episodes,
    select_preview_episodes,
)
from src.evaluate import binary_dice, binary_iou, mean_metrics

HEADLINE_IMAGE_SIZE = 768
HEADLINE_SVD_COMPONENTS = 500


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DOMAIN_DIRS),
        help="Processed 2D domain under --input-dir",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root containing polyp/, kidney_tumor/, cardiac/ 2D caches",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=INSID3_N_EPISODES,
        help=(
            f"Number of random 1-shot episodes (INSID3 lung/ISIC default {INSID3_N_EPISODES})"
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Reference images per episode (INSID3 default 1; only 1-shot is implemented)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="NumPy seed for episode sampling (INSID3 default 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List episodes and exit without importing or running INSID3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Write JSON and preds under --output-dir/<dataset>/",
    )
    parser.add_argument(
        "--preview",
        nargs="?",
        type=int,
        const=PREVIEW_N,
        default=None,
        metavar="N",
        help=(
            f"Randomly pick N episodes from the full sampled list "
            f"(default N={PREVIEW_N}). Headline eval omits this flag."
        ),
    )
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="large")
    parser.add_argument("--image-size", type=int, default=HEADLINE_IMAGE_SIZE)
    parser.add_argument("--svd-comps", type=int, default=HEADLINE_SVD_COMPONENTS)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-thresh", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def dataset_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.dataset


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def serialize_episodes(episodes: list[Episode]) -> list[dict[str, str | int]]:
    return [
        {
            "episode_index": ep.episode_index,
            "dataset": ep.dataset,
            "reference_id": ep.reference_id,
            "target_id": ep.target_id,
            "reference_image": str(ep.reference_image),
            "reference_mask": str(ep.reference_mask),
            "target_image": str(ep.target_image),
            "target_mask": str(ep.target_mask),
        }
        for ep in episodes
    ]


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    episodes = sample_insid3_episodes(
        args.input_dir,
        args.dataset,
        n_episodes=args.episodes,
        n_shots=args.shots,
        seed=args.seed,
    )
    if args.preview is not None:
        if args.preview < 1:
            raise ValueError("--preview must be >= 1")
        episodes = select_preview_episodes(episodes, args.preview, seed=args.seed)
    return episodes


def run_dry_run(args: argparse.Namespace, episodes: list[Episode] | None = None) -> list[Episode]:
    if episodes is None:
        episodes = load_episodes(args)
    payload = serialize_episodes(episodes)
    write_json(dataset_output_dir(args) / "episodes.json", payload)
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return episodes


def _load_binary_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)


def as_binary_mask(pred) -> np.ndarray:
    array = np.asarray(pred.detach().cpu() if hasattr(pred, "detach") else pred)
    binary = (array != 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2D predicted mask, got {binary.shape}")
    if set(np.unique(binary).tolist()) - {0, 1}:
        raise ValueError("predicted mask values must be in {0, 1}")
    return binary


def run_inference(args: argparse.Namespace, episodes: list[Episode] | None = None) -> dict[str, float]:
    if episodes is None:
        episodes = load_episodes(args)
    if not episodes:
        raise ValueError("no episodes to run")

    from src.backbones.dinov3_backbone import build_insid3_model

    model = build_insid3_model(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=args.svd_comps,
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        device=args.device,
    )

    out_dir = dataset_output_dir(args)
    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    per_item: list[dict[str, float | str | int]] = []
    iterator = episodes
    if len(episodes) > 1:
        from tqdm import tqdm

        iterator = tqdm(episodes, desc=args.dataset)
    for ep in iterator:
        try:
            model.set_reference(str(ep.reference_image), str(ep.reference_mask))
            model.set_target(str(ep.target_image))
            pred = as_binary_mask(model.segment())
        finally:
            model.reset_state()
        gt = _load_binary_mask(ep.target_mask)
        if pred.shape != gt.shape:
            raise ValueError(
                f"mask shape mismatch for {ep.target_id}: pred {pred.shape} vs gt {gt.shape}"
            )
        pred_name = f"{ep.episode_index:04d}_{ep.target_id}.png"
        save_binary_mask_png(pred_dir / pred_name, pred)
        iou = binary_iou(pred, gt)
        dice = binary_dice(pred, gt)
        if not math.isfinite(iou) or not math.isfinite(dice):
            raise ValueError(f"non-finite metric for {ep.target_id}: IoU={iou} Dice={dice}")
        pairs.append((pred, gt))
        per_item.append(
            {
                "episode_index": ep.episode_index,
                "reference_id": ep.reference_id,
                "target_id": ep.target_id,
                "IoU": iou,
                "Dice": dice,
            }
        )

    summary = mean_metrics(pairs)
    payload = {
        "dataset": args.dataset,
        "protocol": "insid3_random",
        "seed": args.seed,
        "shots": args.shots,
        "model_size": args.model_size,
        "image_size": args.image_size,
        "svd_comps": args.svd_comps,
        "n": int(summary["n"]),
        "mIoU": summary["mIoU"],
        "Dice": summary["Dice"],
        "items": per_item,
        "pred_dir": str(pred_dir),
    }
    write_json(out_dir / "metrics.json", payload)
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = load_episodes(args)
    if args.dry_run:
        run_dry_run(args, episodes)
        return 0
    run_inference(args, episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
