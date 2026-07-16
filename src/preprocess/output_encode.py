"""Fail-closed SDR output encoding for the current legacy renderer."""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
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


def normalized_icc_profile_sha256(profile: bytes) -> str:
    """Fingerprint ICC semantics while ignoring standard mutable header fields."""
    if len(profile) < 128:
        raise ValueError("ICC profile is shorter than the 128-byte header")
    normalized = bytearray(profile)
    normalized[24:36] = b"\x00" * 12  # profile creation date/time
    normalized[84:100] = b"\x00" * 16  # optional profile ID
    return hashlib.sha256(normalized).hexdigest()


def srgb_icc_profile_fingerprint_sha256() -> str:
    """Return the stable normalized fingerprint of the generated sRGB profile."""
    return normalized_icc_profile_sha256(srgb_icc_profile())


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        image.save(temporary, format_name, icc_profile=srgb_icc_profile(), **options)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            tifffile.imwrite(
                handle,
                encoded,
                photometric="rgb",
                planarconfig="contig",
                metadata=None,
                extratags=[(34675, "B", len(profile), profile, False)],
            )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "TIFF"


def _png_iccp_chunk(profile: bytes) -> bytes:
    chunk_type = b"iCCP"
    payload = b"K-MCFM sRGB\x00\x00" + zlib.compress(profile, level=9)
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def _inject_png_icc(png: bytes, profile: bytes) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    if not png.startswith(signature) or png[12:16] != b"IHDR":
        raise ValueError("encoder returned an invalid PNG stream")
    ihdr_length = struct.unpack(">I", png[8:12])[0]
    ihdr_end = 8 + 12 + ihdr_length
    return png[:ihdr_end] + _png_iccp_chunk(profile) + png[ihdr_end:]


def save_srgb16_png(rgb: np.ndarray, path: Path) -> str:
    """Encode finite HxWx3 display-sRGB values as true uint16 RGB PNG."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("sRGB output must be an HxWx3 array")
    if not np.isfinite(rgb).all():
        raise ValueError("sRGB output contains non-finite values")
    if path.suffix.casefold() != ".png":
        raise ValueError("16-bit PNG output requires a .png extension")
    import cv2

    encoded_rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
    succeeded, buffer = cv2.imencode(
        ".png",
        encoded_rgb[..., ::-1],
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not succeeded:
        raise ValueError("OpenCV failed to encode 16-bit PNG")
    payload = _inject_png_icc(buffer.tobytes(), srgb_icc_profile())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "PNG"
