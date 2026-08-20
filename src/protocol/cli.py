"""Shared CLI flags for method runners."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.constants import DEFAULT_SEED, PREVIEW_N
from src.datasets.episodes import DOMAIN_DIRS
from src.protocol.episodes_io import load_persisted_episodes, maybe_preview


def add_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: str,
    default_image_size: int,
) -> argparse.ArgumentParser:
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
        "--output-dir",
        type=Path,
        default=Path(default_output),
        help="Method root. The runner appends --dataset.",
    )
    parser.add_argument(
        "--episodes-json",
        type=Path,
        default=None,
        help="Phase 3 episodes.json. Default: results/<dataset>/episodes.json",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preview",
        nargs="?",
        type=int,
        const=PREVIEW_N,
        default=None,
        metavar="N",
        help=f"Randomly pick N persisted episodes (default N={PREVIEW_N})",
    )
    parser.add_argument("--image-size", type=int, default=default_image_size)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if metrics.json already matches this protocol",
    )
    return parser


def resolve_device(device: str) -> str:
    if device not in (None, "auto"):
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_episodes_json(args: argparse.Namespace) -> Path:
    if args.episodes_json is not None:
        return Path(args.episodes_json)
    return Path("results") / args.dataset / "episodes.json"


def load_eval_episodes(args: argparse.Namespace):
    path = resolve_episodes_json(args)
    if not path.is_file():
        raise SystemExit(
            f"episodes JSON missing: {path}. "
            "Run Phase 3 dry-run first, or pass --episodes-json."
        )
    episodes = load_persisted_episodes(path, input_dir=args.input_dir, dataset=args.dataset)
    return maybe_preview(episodes, args.preview, seed=args.seed)


def method_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.dataset
