"""Raster image inspection and SDR decode helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .types import DecodeWarning, InputInspection, SourceProfile, WorkingImage


_MODE_BIT_DEPTH = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
    "I": 32,
    "F": 32,
}


def _orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    value = exif.get(274)
    return int(value) if value else None


def _source_profile(image: Image.Image) -> SourceProfile:
    icc = image.info.get("icc_profile")
    if icc:
        return SourceProfile("icc", "embedded ICC profile", len(icc))
    if image.format in {"HEIF", "HEIC", "AVIF"}:
        return SourceProfile("nclx", "container color metadata may be present; backend support limited")
    return SourceProfile("assumed_srgb", "no embedded ICC profile; assuming sRGB")


def _hdr_metadata(image: Image.Image) -> dict[str, Any]:
    keys = {}
    for key, value in image.info.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("hdr", "gain", "cicp", "nclx", "mastering", "icc")):
            if isinstance(value, bytes):
                keys[key] = f"bytes:{len(value)}"
            else:
                keys[key] = str(value)
    return keys


def inspect_raster(path: Path) -> InputInspection:
    warnings: list[DecodeWarning] = []
    try:
        with Image.open(path) as image:
            profile = _source_profile(image)
            if profile.kind == "assumed_srgb":
                warnings.append(DecodeWarning("assumed_srgb", profile.description))
            if image.format in {"HEIF", "HEIC", "AVIF"}:
                warnings.append(
                    DecodeWarning(
                        "limited_heif_hdr",
                        "HEIF/AVIF HDR or gain-map reconstruction is not implemented in this backend.",
                    )
                )
            return InputInspection(
                path=path,
                exists=True,
                source_kind="raster",
                format_name=image.format or "unknown",
                mode=image.mode,
                width=image.width,
                height=image.height,
                bit_depth=_MODE_BIT_DEPTH.get(image.mode),
                has_alpha=image.mode in {"LA", "RGBA"} or "transparency" in image.info,
                orientation=_orientation(image),
                frame_count=getattr(image, "n_frames", 1),
                source_profile=profile,
                transfer_state="display_referred",
                hdr_metadata=_hdr_metadata(image),
                warnings=warnings,
            )
    except UnidentifiedImageError:
        return InputInspection(path=path, exists=path.exists(), source_kind="unknown")


def _convert_with_icc(image: Image.Image, warnings: list[DecodeWarning]) -> Image.Image:
    icc = image.info.get("icc_profile")
    if not icc:
        warnings.append(DecodeWarning("assumed_srgb", "No ICC profile; interpreting raster RGB as sRGB."))
        return image.convert("RGB")
    try:
        src = ImageCms.ImageCmsProfile(BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(image.convert("RGB"), src, dst, outputMode="RGB")
        return converted
    except Exception as exc:  # noqa: BLE001 - fallback should stay non-fatal.
        warnings.append(DecodeWarning("icc_convert_failed", f"ICC conversion failed; used RGB fallback: {exc!r}"))
        return image.convert("RGB")


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def working_image_to_legacy_srgb8(working: WorkingImage) -> Image.Image:
    """Explicit temporary adapter from WorkingImage to the 8-bit legacy renderer."""
    if working.working_space != "linear_srgb" or working.transfer_state != "display_linear":
        raise ValueError(
            "legacy adapter requires linear_srgb/display_linear WorkingImage; "
            f"got {working.working_space}/{working.transfer_state}"
        )
    clipped = np.clip(working.pixels, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return Image.fromarray(np.rint(encoded * 255.0).astype(np.uint8), mode="RGB")


def load_raster_working_image(path: Path) -> WorkingImage:
    inspection = inspect_raster(path)
    warnings = list(inspection.warnings)
    if inspection.source_kind != "raster":
        raise ValueError(f"Unsupported raster input: {path}")
    with Image.open(path) as raw_image:
        oriented = ImageOps.exif_transpose(raw_image)
        alpha_policy = "absent"
        if oriented.mode in {"LA", "RGBA"}:
            alpha_policy = "preserved"
        rgb_image = _convert_with_icc(oriented, warnings)
        arr = np.asarray(rgb_image, dtype=np.float32) / 255.0
    pixels = _srgb_to_linear(np.clip(arr, 0.0, 1.0))
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_profile=inspection.source_profile,
        hdr_metadata=inspection.hdr_metadata,
        orientation_applied=True,
        alpha_policy=alpha_policy,
        bit_depth_in=inspection.bit_depth,
        source_path=path,
        warnings=warnings,
    )
