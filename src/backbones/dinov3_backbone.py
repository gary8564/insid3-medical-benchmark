"""Load frozen DINOv3 through INSID3 with absolute weight paths."""

from __future__ import annotations

import sys
from pathlib import Path

from src.data.paths import INSID3_ROOT, PRETRAIN_ROOT

HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

DINOV3_HUB = "facebookresearch/dinov3"

WEIGHT_FILES = {
    "small": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}

# DINOv3 ViT-S/B/L channel counts. SVD rank cannot exceed min(C, n_patches).
_ENCODER_CHANNELS = {"small": 384, "base": 768, "large": 1024}

PATCH_SIZE = 16


def insid3_repo_present() -> bool:
    return (INSID3_ROOT / "models" / "insid3.py").is_file()


def weight_path(model_size: str) -> Path:
    if model_size not in WEIGHT_FILES:
        raise KeyError(f"unknown model_size {model_size!r}; expected one of {sorted(WEIGHT_FILES)}")
    return PRETRAIN_ROOT / WEIGHT_FILES[model_size]


def weights_available(model_size: str) -> bool:
    return weight_path(model_size).is_file() and weight_path(model_size).stat().st_size > 0


def clamp_svd_components(model_size: str, image_size: int, svd_components: int) -> int:
    n_patches = (image_size // PATCH_SIZE) ** 2
    limit = min(_ENCODER_CHANNELS[model_size], n_patches)
    return max(1, min(int(svd_components), limit))


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ensure_insid3_repo() -> Path:
    if insid3_repo_present():
        return INSID3_ROOT
    raise FileNotFoundError(
        f"INSID3 submodule is missing at {INSID3_ROOT}. "
        "From the repo root run: git submodule update --init"
    )


def add_insid3_to_syspath(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else INSID3_ROOT
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return root


def _dinov3_hub_repo() -> Path:
    """
    Download or reuse the `facebookresearch/dinov3` source tree under torch hub cache.
    INSID3 needs DINOv3's Python package (`dinov3.hub.backbones`) to construct the ViT, then loads weights from our `pretrain/` files. 
    """
    import torch

    return Path(
        torch.hub._get_cache_or_reload(
            DINOV3_HUB,
            force_reload=False,
            trust_repo=True,
            verbose=True,
            skip_validation=False,
        )
    )


def load_dinov3_encoder(model_size: str, weights: Path):
    """Build a frozen DINOv3 ViT from a local `.pth`, importing only the backbone API."""
    from importlib import import_module

    repo = str(_dinov3_hub_repo().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    builder = getattr(import_module("dinov3.hub.backbones"), HUB_NAMES[model_size])
    return builder(weights=str(weights.resolve()))


def can_run_insid3(model_size: str = "small") -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return insid3_repo_present() and weights_available(model_size)


def build_insid3_model(
    *,
    model_size: str = "small",
    image_size: int = 256,
    svd_components: int = 64,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    device: str | None = None,
):
    """Construct INSID3 with local DINOv3 weights (no CRF)."""
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

    weights = weight_path(model_size)
    if not weights.is_file():
        raise SystemExit(
            f"DINOv3 weights missing: {weights}. "
            "Request access and place the checkpoint under pretrain/. See docs/data.md."
        )

    from models.insid3 import INSID3

    resolved_device = auto_device() if device in (None, "auto") else device
    svd = clamp_svd_components(model_size, image_size, svd_components)
    encoder = load_dinov3_encoder(model_size, weights)
    model = INSID3(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd,
        tau=tau,
        merge_threshold=merge_threshold,
        mask_refiner="bilinear",
        resize_to_orig_size=True,
        device=resolved_device,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model.to(resolved_device)
