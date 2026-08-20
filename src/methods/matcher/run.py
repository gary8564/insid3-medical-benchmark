"""Matcher CLI. ``python -m src.methods.matcher``."""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from src.data.paths import DINOV2_VITL14_WEIGHTS, SAM_VIT_H_WEIGHTS
from src.datasets.episodes import Episode
from src.protocol.cli import add_eval_args, load_eval_episodes, method_output_dir, resolve_device
from src.protocol.fss_tensors import load_mask_float, load_rgb_01, pack_query, pack_supports
from src.protocol.loop import run_episodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_args(parser, default_output="results/matcher", default_image_size=518)
    parser.add_argument("--dinov2-weights", type=str, default=str(DINOV2_VITL14_WEIGHTS))
    parser.add_argument("--sam-weights", type=str, default=str(SAM_VIT_H_WEIGHTS))
    return parser.parse_args(argv)


def _predict_factory(model, image_size: int, device: str):
    def predict(ep: Episode) -> np.ndarray:
        images = [load_rgb_01(path, image_size).to(device) for path in ep.reference_images]
        masks = [load_mask_float(path, image_size).to(device) for path in ep.reference_masks]
        if any(mask.sum().item() == 0 for mask in masks):
            warnings.warn(
                f"empty reference mask in episode {ep.episode_index}; "
                "Matcher will punch a 14×14 centre square",
                stacklevel=2,
            )
        support_imgs, support_masks = pack_supports(images, masks)
        query = pack_query(load_rgb_01(ep.target_image, image_size).to(device))
        try:
            model.set_reference(support_imgs, support_masks)
            model.set_target(query)
            pred_masks = model.predict()
        finally:
            model.clear()
        pred = pred_masks.detach().cpu().numpy() if hasattr(pred_masks, "detach") else np.asarray(pred_masks)
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(f"Matcher pred has shape {pred.shape}")
        return (pred > 0).astype(np.uint8)

    return predict


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = load_eval_episodes(args)
    device = resolve_device(args.device)
    from src.methods.matcher.build import build_matcher

    model = build_matcher(
        dinov2_weights=args.dinov2_weights,
        sam_weights=args.sam_weights,
        image_size=args.image_size,
        device=device,
    )
    run_episodes(
        episodes,
        _predict_factory(model, args.image_size, device),
        out_dir=method_output_dir(args),
        metrics_fields={
            "dataset": args.dataset,
            "protocol": "insid3_random",
            "seed": args.seed,
            "shots": 1,
            "method": "matcher",
            "image_size": args.image_size,
            "prompt": "1shot_mask",
            "tier": "B",
        },
        overwrite=args.overwrite,
        isolate_errors=True,
    )
    return 0
