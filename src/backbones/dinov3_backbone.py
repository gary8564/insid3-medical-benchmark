"""Re-export. Encoder loading lives in ``src.methods.insid3``."""

from src.methods.insid3.build import build_insid3_model
from src.methods.insid3.dinov3 import (
    DINOV3_HUB,
    HUB_NAMES,
    PATCH_SIZE,
    WEIGHT_FILES,
    add_insid3_to_syspath,
    auto_device,
    can_run_insid3,
    clamp_svd_components,
    ensure_insid3_repo,
    insid3_repo_present,
    load_dinov3_encoder,
    weight_path,
    weights_available,
)

__all__ = [
    "DINOV3_HUB",
    "HUB_NAMES",
    "PATCH_SIZE",
    "WEIGHT_FILES",
    "add_insid3_to_syspath",
    "auto_device",
    "build_insid3_model",
    "can_run_insid3",
    "clamp_svd_components",
    "ensure_insid3_repo",
    "insid3_repo_present",
    "load_dinov3_encoder",
    "weight_path",
    "weights_available",
]
