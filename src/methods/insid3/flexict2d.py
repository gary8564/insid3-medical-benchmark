"""FlexiCT-2D encoder adapter + INSID3 debias switch. Do not edit the INSID3 submodule."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.data.paths import FLEXICT_ROOT
from src.methods.insid3.dinov3 import add_insid3_to_syspath, ensure_insid3_repo

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_NORM = "png_uint8_to_minus1_1"


def add_flexict_to_syspath() -> Path:
    if not (FLEXICT_ROOT / "flexi_ct" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"FlexiCT submodule is missing at {FLEXICT_ROOT}. "
            "From the repo root run: git submodule update --init --recursive"
        )
    resolved = str(FLEXICT_ROOT.resolve())
    import sys

    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return FLEXICT_ROOT


def imagenet3_to_flexict1(x: torch.Tensor) -> torch.Tensor:
    """Undo INSID3 ImageNet RGB, average to 1 channel, map to FlexiCT [-1, 1]."""
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb01 = (x * std + mean).clamp(0.0, 1.0)
    gray01 = rgb01.mean(dim=1, keepdim=True)
    return gray01 * 2.0 - 1.0


class FlexiCT2DEncoder(nn.Module):
    """Expose ``get_intermediate_layers`` the way INSID3 calls it."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        patch_size: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        add_flexict_to_syspath()
        from flexi_ct import Flexi_CT_2D

        self.model = Flexi_CT_2D(checkpoint_path=str(checkpoint_path), device=device)
        self.patch_size = int(patch_size)
        if self.patch_size != 8:
            self.model.backbone.patch_embed_2D.set_patch_size(self.patch_size)
            self.model.backbone.patch_size = self.patch_size

    def get_intermediate_layers(self, x, n=1, reshape=True):
        x1 = imagenet3_to_flexict1(x)
        return self.model.backbone.get_intermediate_layers(x1, n=n, reshape=reshape)


def _insid3_class():
    ensure_insid3_repo()
    add_insid3_to_syspath()
    from models.insid3 import INSID3

    return INSID3


class INSID3DebiasSwitch:
    """Built after INSID3 is on ``sys.path``. Subclasses ``INSID3._debias_features``."""

    def __new__(cls, *args, **kwargs):
        base = _insid3_class()

        class _Switch(base):
            def __init__(self, *a, use_debiased: bool = True, **kw):
                self.use_debiased = use_debiased
                super().__init__(*a, **kw)

            def _debias_features(self, fmaps_norm):
                if not self.use_debiased:
                    return fmaps_norm
                return super()._debias_features(fmaps_norm)

        return _Switch(*args, **kwargs)
