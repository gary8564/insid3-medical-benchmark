"""Download unmodified ACDC training NIfTI (ED/ES frames + masks, no 4D cine)."""

from __future__ import annotations

from pathlib import Path

from src.data.paths import ACDC_RAW, ensure_dir

ACDC_REPO = "MedOtter/ACDC"
# Training patients only (Info.cfg + ED/ES frames + gt). Skip unlabeled testing/
# and 4D cine volumes.
ACDC_PATTERNS = (
    "training/*/Info.cfg",
    "training/*/*frame*.nii.gz",
    "training/MANDATORY_CITATION.md",
)
ACDC_IGNORE = (
    "testing/**",
    "**/*_4d.nii.gz",
)


def acdc_is_ready(raw_root: Path) -> bool:
    training = raw_root / "training"
    if not training.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("patient") for p in training.iterdir())


def download_acdc(raw_root: Path | None = None) -> Path:
    dest = Path(raw_root) if raw_root is not None else ACDC_RAW
    ensure_dir(dest)
    if acdc_is_ready(dest):
        return dest
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=ACDC_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=list(ACDC_PATTERNS),
        ignore_patterns=list(ACDC_IGNORE),
    )
    return dest


if __name__ == "__main__":
    download_acdc()
