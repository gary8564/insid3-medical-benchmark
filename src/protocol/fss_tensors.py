"""PNG → Matcher / GF-SAM tensors. Images are float [0, 1], not ImageNet-normalised."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_01(path: Path | str, size: int):
    """PIL RGB, bilinear resize → ``(3, size, size)`` float32 in [0, 1]."""
    import torch

    image = Image.open(path).convert("RGB").resize((size, size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_mask_float(path: Path | str, size: int):
    """Nearest resize → ``(size, size)`` float32 in {0, 1}."""
    import torch

    mask = Image.open(path).convert("L").resize((size, size), resample=Image.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.float32)
    return torch.from_numpy(array)


def pack_support(img, mask):
    """``img`` (3, H, W), ``mask`` (H, W) → batch-1 one-shot FSS tensors."""
    support_imgs = img.unsqueeze(0).unsqueeze(0)
    support_masks = mask.unsqueeze(0).unsqueeze(0)
    return support_imgs, support_masks


def pack_supports(images: list, masks: list):
    """k-shot: ``(1, nshot, 3, H, W)`` and ``(1, nshot, H, W)``."""
    import torch

    if not images:
        raise ValueError("need at least one support image")
    support_imgs = torch.stack(images, dim=0).unsqueeze(0)
    support_masks = torch.stack(masks, dim=0).unsqueeze(0)
    return support_imgs, support_masks


def pack_query(img):
    """``img`` (3, H, W) → ``(1, 3, H, W)``."""
    return img.unsqueeze(0)
