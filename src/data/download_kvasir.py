"""Download Kvasir-SEG from Simula."""

from __future__ import annotations

from pathlib import Path

from src.data.io_utils import download_url, find_named_dir, unzip
from src.data.paths import KVASIR_RAW, ensure_dir

# Full Kvasir-SEG zip: 1,000 RGB images with paired binary polyp masks.
KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def kvasir_is_ready(raw_root: Path) -> bool:
    try:
        find_named_dir(raw_root, "images")
        find_named_dir(raw_root, "masks")
        return True
    except FileNotFoundError:
        return False


def download_kvasir(raw_root: Path | None = None) -> Path:
    dest = Path(raw_root) if raw_root is not None else KVASIR_RAW
    ensure_dir(dest)
    if kvasir_is_ready(dest):
        return dest
    archive = dest / "kvasir-seg.zip"
    download_url(KVASIR_URL, archive)
    unzip(archive, dest)
    return dest


if __name__ == "__main__":
    download_kvasir()
