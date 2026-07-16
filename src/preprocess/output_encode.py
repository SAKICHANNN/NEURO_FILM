"""Fail-closed SDR output encoding for the current legacy renderer."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms


_OUTPUT_FORMATS: dict[str, tuple[str, dict[str, object]]] = {
    ".png": ("PNG", {}),
    ".jpg": ("JPEG", {"quality": 95, "subsampling": 0}),
    ".jpeg": ("JPEG", {"quality": 95, "subsampling": 0}),
    ".tif": ("TIFF", {}),
    ".tiff": ("TIFF", {}),
}


@lru_cache(maxsize=1)
def srgb_icc_profile() -> bytes:
    """Return the LittleCMS-generated standard sRGB output profile."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def srgb_icc_profile_sha256() -> str:
    """Identify the exact generated profile bytes for replay provenance."""
    return hashlib.sha256(srgb_icc_profile()).hexdigest()


def save_srgb8(rgb: np.ndarray, path: Path) -> str:
    """Encode finite HxWx3 display-sRGB values according to the file extension."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("sRGB output must be an HxWx3 array")
    if not np.isfinite(rgb).all():
        raise ValueError("sRGB output contains non-finite values")
    try:
        format_name, options = _OUTPUT_FORMATS[path.suffix.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported output extension: {path.suffix or '<none>'}") from exc
    encoded = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(encoded, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format_name, icc_profile=srgb_icc_profile(), **options)
    return format_name


def save_srgb16_tiff(rgb: np.ndarray, path: Path) -> str:
    """Encode finite HxWx3 display-sRGB values as true uint16 RGB TIFF."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("sRGB output must be an HxWx3 array")
    if not np.isfinite(rgb).all():
        raise ValueError("sRGB output contains non-finite values")
    if path.suffix.casefold() not in {".tif", ".tiff"}:
        raise ValueError("16-bit TIFF output requires a .tif or .tiff extension")
    encoded = np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
    profile = srgb_icc_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        encoded,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    return "TIFF"
