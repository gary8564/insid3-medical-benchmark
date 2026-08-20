"""Machine / install provenance for ``run.json`` (not scores)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.data.paths import REPO_ROOT, THIRD_PARTY_ROOT

_SUBMODULE_DIRS = {
    "INSID3": "INSID3",
    "GF-SAM": "GF-SAM",
    "Matcher": "Matcher",
    "FlexiCT": "FlexiCT",
    "MedSAM2": "MedSAM2",
}


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:
        return None
    try:
        return version(name)
    except Exception:
        return None


def _git_head(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _cuda_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def collect_run_meta() -> dict[str, object]:
    submodules = {}
    for name, folder in _SUBMODULE_DIRS.items():
        root = THIRD_PARTY_ROOT / folder
        submodules[name] = _git_head(root) if root.is_dir() else None
    return {
        "python": sys.version.split()[0],
        "torch": _pkg_version("torch"),
        "torchvision": _pkg_version("torchvision"),
        "numpy": _pkg_version("numpy"),
        "cuda": _cuda_name(),
        "git_head": _git_head(REPO_ROOT),
        "submodules": submodules,
    }
