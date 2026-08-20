"""Build INSID3 with DINOv3 or FlexiCT-2D. No CRF."""

from __future__ import annotations

from pathlib import Path

from src.data.paths import FLEXICT_2D_WEIGHTS
from src.methods.insid3.dinov3 import (
    add_insid3_to_syspath,
    auto_device,
    clamp_svd_components,
    ensure_insid3_repo,
    load_dinov3_encoder,
    weight_path,
)


def build_insid3_model(
    *,
    model_size: str = "small",
    image_size: int = 256,
    svd_components: int = 64,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    device: str | None = None,
    backbone: str = "dinov3",
    flexict_weights: Path | None = None,
    flexict_patch_size: int = 16,
    debiased: bool = True,
):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required for INSID3 inference. Install with: uv sync --extra torch --group dev"
        ) from exc

    try:
        root = ensure_insid3_repo()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    add_insid3_to_syspath(root)

    resolved_device = auto_device() if device in (None, "auto") else device
    if backbone == "flexict2d":
        from src.methods.insid3.flexict2d import FlexiCT2DEncoder, INSID3DebiasSwitch

        weights = Path(flexict_weights) if flexict_weights is not None else FLEXICT_2D_WEIGHTS
        if not weights.is_file():
            raise SystemExit(
                f"FlexiCT-2D weights missing: {weights}. "
                "Download the 2D teacher checkpoint to pretrain/flexict_2d_teacher.pth."
            )
        encoder = FlexiCT2DEncoder(
            checkpoint_path=weights,
            patch_size=flexict_patch_size,
            device=resolved_device,
        )
        svd = min(int(svd_components), 864)
        model = INSID3DebiasSwitch(
            encoder=encoder,
            image_size=image_size,
            svd_components=svd,
            tau=tau,
            merge_threshold=merge_threshold,
            mask_refiner="bilinear",
            resize_to_orig_size=True,
            device=resolved_device,
            use_debiased=debiased,
        )
    elif backbone == "dinov3":
        from src.methods.insid3.flexict2d import INSID3DebiasSwitch

        weights = weight_path(model_size)
        if not weights.is_file():
            raise SystemExit(
                f"DINOv3 weights missing: {weights}. "
                "Request access and place the checkpoint under pretrain/. See docs/data.md."
            )
        svd = clamp_svd_components(model_size, image_size, svd_components)
        encoder = load_dinov3_encoder(model_size, weights)
        model = INSID3DebiasSwitch(
            encoder=encoder,
            image_size=image_size,
            svd_components=svd,
            tau=tau,
            merge_threshold=merge_threshold,
            mask_refiner="bilinear",
            resize_to_orig_size=True,
            device=resolved_device,
            use_debiased=debiased,
        )
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model.to(resolved_device)
