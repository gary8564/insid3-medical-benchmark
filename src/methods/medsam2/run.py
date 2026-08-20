"""Box-prompted MedSAM2 on our 2D slices. ``python -m src.methods.medsam2``."""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from src.data.paths import MEDSAM2_WEIGHTS
from src.datasets.episodes import Episode
from src.methods.medsam2.box import mask2D_to_bbox
from src.methods.medsam2.build import DEFAULT_CFG
from src.protocol.cli import add_eval_args, load_eval_episodes, method_output_dir, resolve_device
from src.protocol.loop import run_episodes
from src.protocol.masks import load_binary_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CANVAS = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_args(parser, default_output="results/medsam2", default_image_size=CANVAS)
    parser.add_argument("--checkpoint", type=str, default=str(MEDSAM2_WEIGHTS))
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    return parser.parse_args(argv)


def _prepare_frame(path, device):
    import torch

    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size
    resized = np.asarray(image.resize((CANVAS, CANVAS), resample=Image.BILINEAR), dtype=np.float32)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy() / 255.0).to(device)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), orig_h, orig_w


def _predict_factory(predictor, device: str):
    import torch

    def predict(ep: Episode) -> np.ndarray:
        gt = load_binary_mask(ep.target_mask)
        box = mask2D_to_bbox(gt, max_shift=0)
        if box is None:
            return np.zeros(gt.shape, dtype=np.uint8)
        img, orig_h, orig_w = _prepare_frame(ep.target_image, device)
        state = predictor.init_state(img, orig_h, orig_w)
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, _, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=box,
                )
            pred = (out_mask_logits[0] > 0.0).detach().cpu().numpy()
            pred = np.squeeze(pred)
            if pred.ndim != 2:
                raise ValueError(f"MedSAM2 pred has shape {pred.shape}")
            return pred.astype(np.uint8)
        finally:
            predictor.reset_state(state)

    return predict


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = load_eval_episodes(args)
    device = resolve_device(args.device)
    if not str(device).startswith("cuda"):
        raise SystemExit("MedSAM2 requires a GPU (--device cuda). Do not CPU-eval 600 episodes.")
    from src.methods.medsam2.build import build_medsam2_predictor

    predictor = build_medsam2_predictor(checkpoint=args.checkpoint, cfg=args.cfg)
    run_episodes(
        episodes,
        _predict_factory(predictor, device),
        out_dir=method_output_dir(args),
        metrics_fields={
            "dataset": args.dataset,
            "protocol": "insid3_random",
            "seed": args.seed,
            "shots": 1,
            "method": "medsam2",
            "image_size": CANVAS,
            "prompt": "box_from_gt",
            "tier": "C",
        },
        overwrite=args.overwrite,
    )
    return 0
