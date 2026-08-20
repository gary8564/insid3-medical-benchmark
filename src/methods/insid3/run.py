"""INSID3 CLI. ``python src/run_insid3.py`` is a shim so the Phase 3 notebook still works."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.constants import (
    DEFAULT_SEED,
    DEFAULT_SHOTS,
    INSID3_N_EPISODES,
    PREVIEW_N,
)
from src.data.paths import FLEXICT_2D_WEIGHTS
from src.datasets.episodes import DOMAIN_DIRS, Episode, sample_insid3_episodes
from src.protocol.episodes_io import (
    load_persisted_episodes,
    maybe_preview,
    serialize_episodes,
    write_json,
)
from src.protocol.loop import run_episodes

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
        help=f"Number of random episodes when not using --episodes-json (default {INSID3_N_EPISODES})",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Same-class reference images per episode (INSID3 default 1).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List episodes and exit without importing or running INSID3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write JSON and preds here. Default: results/<dataset>/",
    )
    parser.add_argument(
        "--preview",
        nargs="?",
        type=int,
        const=PREVIEW_N,
        default=None,
        metavar="N",
        help=f"Randomly pick N episodes (default N={PREVIEW_N}). Headline eval omits this flag.",
    )
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="large")
    parser.add_argument("--image-size", type=int, default=HEADLINE_IMAGE_SIZE)
    parser.add_argument("--svd-comps", type=int, default=HEADLINE_SVD_COMPONENTS)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-thresh", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backbone",
        choices=("dinov3", "flexict2d"),
        default="dinov3",
    )
    parser.add_argument("--flexict-weights", type=Path, default=FLEXICT_2D_WEIGHTS)
    parser.add_argument("--flexict-patch-size", type=int, choices=(8, 16), default=16)
    parser.add_argument(
        "--debiased",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="INSID3 positional debiasing (default on). --no-debiased is the Phase 7 ablation.",
    )
    parser.add_argument(
        "--episodes-json",
        type=Path,
        default=None,
        help="Reuse a Phase 3 episodes.json instead of sampling. Required for --backbone flexict2d.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def dataset_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is None:
        if args.backbone == "flexict2d":
            condition = "debiased" if args.debiased else "nodebiased"
            return Path("results") / "flexict2d" / condition / args.dataset
        return Path("results") / args.dataset
    if args.backbone == "flexict2d":
        return Path(args.output_dir) / args.dataset
    return Path(args.output_dir)


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.backbone == "flexict2d" and args.dataset == "polyp":
        raise SystemExit("FlexiCT-2D on polyp is out of scope (CT-pretrained encoder).")
    if args.episodes_json is not None:
        episodes = load_persisted_episodes(
            args.episodes_json, input_dir=args.input_dir, dataset=args.dataset
        )
        return maybe_preview(episodes, args.preview, seed=args.seed)
    if args.backbone == "flexict2d":
        raise SystemExit("--episodes-json is required when --backbone flexict2d (do not re-sample).")
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    episodes = sample_insid3_episodes(
        args.input_dir,
        args.dataset,
        n_episodes=args.episodes,
        n_shots=args.shots,
        seed=args.seed,
    )
    return maybe_preview(episodes, args.preview, seed=args.seed)


def run_dry_run(args: argparse.Namespace, episodes: list[Episode] | None = None) -> list[Episode]:
    if episodes is None:
        episodes = load_episodes(args)
    payload = serialize_episodes(episodes)
    write_json(dataset_output_dir(args) / "episodes.json", payload)
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return episodes


def run_inference(args: argparse.Namespace, episodes: list[Episode] | None = None) -> dict[str, object]:
    if episodes is None:
        episodes = load_episodes(args)
    if not episodes:
        raise ValueError("no episodes to run")

    from src.methods.insid3.build import build_insid3_model

    model = build_insid3_model(
        model_size=args.model_size,
        image_size=args.image_size,
        svd_components=args.svd_comps,
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        device=args.device,
        backbone=args.backbone,
        flexict_weights=args.flexict_weights,
        flexict_patch_size=args.flexict_patch_size,
        debiased=args.debiased,
    )

    def predict(ep: Episode):
        try:
            for image, mask in zip(ep.reference_images, ep.reference_masks, strict=True):
                model.set_reference(str(image), str(mask))
            model.set_target(str(ep.target_image))
            return model.segment()
        finally:
            model.reset_state()

    fields: dict[str, object] = {
        "dataset": args.dataset,
        "protocol": "insid3_random",
        "seed": args.seed,
        "shots": args.shots,
        "model_size": args.model_size if args.backbone == "dinov3" else None,
        "image_size": args.image_size,
        "svd_comps": args.svd_comps,
        "method": "insid3",
        "backbone": args.backbone,
        "debiased": bool(args.debiased),
        "prompt": "1shot_mask",
        "tier": "A",
    }
    if args.backbone == "flexict2d":
        fields["flexict_patch_size"] = args.flexict_patch_size
        fields["input_norm"] = "png_uint8_to_minus1_1"
    return run_episodes(
        episodes,
        predict,
        out_dir=dataset_output_dir(args),
        metrics_fields=fields,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = load_episodes(args)
    if args.dry_run:
        run_dry_run(args, episodes)
        return 0
    run_inference(args, episodes)
    return 0
