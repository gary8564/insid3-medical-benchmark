"""Download, archive, and NIfTI helpers"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

import certifi
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.paths import ensure_dir


def download_url(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    if curl is not None:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "--progress-bar", "-o", str(tmp), url],
            check=True,
        )
    else:
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(url, context=context) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def unzip(archive: Path, dest: Path) -> Path:
    dest = ensure_dir(Path(dest))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


def find_named_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no {name!r} directory under {root}")
    raise FileNotFoundError(f"multiple {name!r} directories under {root}: {matches}")


def slice_to_uint8(plane: np.ndarray) -> np.ndarray:
    """Per-slice 1-99 percentile stretch to 8-bit."""
    values = np.asarray(plane, dtype=np.float32)
    lo, hi = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def save_gray_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(path)


def save_binary_mask_png(path: Path, binary: np.ndarray) -> None:
    save_gray_png(path, (np.asarray(binary) != 0).astype(np.uint8) * 255)


def _require_3d(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"expected 3D volume, got {array.shape}")
    return array


def _smallest_axis(shape: tuple[int, ...]) -> int:
    return int(np.argmin(shape))


# Anatomical planes from the NIfTI affine (RAS+ world axes).
# "acquisition" is the scanned through-plane (thickest spacing), used when
# the series is oblique (e.g. cardiac short-axis) or not a body-axis reformatted.
ANATOMICAL_PLANE_CODES = {
    "axial": frozenset({"S", "I"}),
    "coronal": frozenset({"A", "P"}),
    "sagittal": frozenset({"R", "L"}),
}
SLICE_PLANES = frozenset({*ANATOMICAL_PLANE_CODES, "acquisition"})


def infer_slice_axis(
    volume: np.ndarray,
    affine: np.ndarray | None = None,
    *,
    plane: str,
) -> int:
    """
    Determine the slice axis of the given volume.

    `plane` is required so each dataset names the 2D plane it wants:
    - "axial" / "coronal" / "sagittal": anatomical plane from the affine (superior-inferior, anterior-posterior, left-right).
    - "acquisition": through-plane of the scanned stack (uniquely thickest voxel spacing). Use this for oblique series such as cardiac short-axis (SAX).
    If the affine is missing or the chosen cue is a tie, fall back to the smallest array axis.
    """
    array = _require_3d(volume)
    if plane not in SLICE_PLANES:
        raise ValueError(
            f"plane must be one of {sorted(SLICE_PLANES)}, got {plane!r}"
        )
    if affine is None:
        return _smallest_axis(array.shape)

    affine_arr = np.asarray(affine, dtype=np.float64)
    if affine_arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got {affine_arr.shape}")

    import nibabel as nib

    if plane in ANATOMICAL_PLANE_CODES:
        wanted = ANATOMICAL_PLANE_CODES[plane]
        for axis, code in enumerate(nib.aff2axcodes(affine_arr)):
            if code in wanted:
                return int(axis)
        return _smallest_axis(array.shape)

    spacings = np.asarray(nib.affines.voxel_sizes(affine_arr), dtype=np.float64)
    if spacings.shape != (3,) or not np.all(np.isfinite(spacings)) or np.any(spacings <= 0):
        return _smallest_axis(array.shape)
    order = np.argsort(spacings)
    thicker, thinner = float(spacings[order[-1]]), float(spacings[order[-2]])
    if thicker > thinner * 1.1:
        return int(order[-1])
    return _smallest_axis(array.shape)


def slice_axis_for_pair(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    image: np.ndarray,
    image_affine: np.ndarray,
    *,
    plane: str,
    label: str = "volume",
) -> int:
    """Same infer_slice_axis on image and mask; error if they disagree."""
    mask_axis = infer_slice_axis(mask, mask_affine, plane=plane)
    image_axis = infer_slice_axis(image, image_affine, plane=plane)
    if image_axis != mask_axis:
        raise ValueError(
            f"{label}: image slice axis {image_axis} vs mask slice axis {mask_axis}"
        )
    return mask_axis


def move_slice_axis_first(volume: np.ndarray, slice_axis: int | None = None) -> np.ndarray:
    """
    Permute a 3D volume so the slice stack is axis 0: (n_slices, H, W).

    `slice_axis` is the current stack axis (from `infer_slice_axis`). 
    If omitted, the smallest array axis is moved to the front.
    """
    if slice_axis is None:
        axis = _smallest_axis(_require_3d(volume).shape)
    else:
        axis = slice_axis
    return np.moveaxis(np.asarray(volume), axis, 0)


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI array and corresponding 4x4 voxel-to-world affine matrix."""
    import nibabel as nib

    img = nib.load(str(path))
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 affine, got {affine.shape}")
    return np.asanyarray(img.dataobj), affine
