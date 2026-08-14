"""Download KiPA22 training volumes from Hugging Face."""

from __future__ import annotations

from pathlib import Path

from src.data.io_utils import find_named_dir, unzip
from src.data.paths import KIPA_RAW, ensure_dir

KIPA_REPO = "YongchengYAO/KiPA22"
# Training volumes only (image + label). Open/closed test cases have no public GT.
KIPA_ZIP = "train.zip"


def kipa_is_ready(raw_root: Path) -> bool:
    try:
        find_named_dir(raw_root, "image")
        find_named_dir(raw_root, "label")
        return True
    except FileNotFoundError:
        return False


def download_kipa(raw_root: Path | None = None) -> Path:
    dest = Path(raw_root) if raw_root is not None else KIPA_RAW
    ensure_dir(dest)
    if kipa_is_ready(dest):
        return dest
    archive = dest / KIPA_ZIP
    if not archive.is_file():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=KIPA_REPO,
            filename=KIPA_ZIP,
            repo_type="dataset",
            local_dir=str(dest),
        )
        archive = Path(downloaded)
    unzip(archive, dest)
    return dest


if __name__ == "__main__":
    download_kipa()
