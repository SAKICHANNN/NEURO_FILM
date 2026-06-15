"""RAW inspection and decode helpers based on rawpy/LibRaw."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import DecodeWarning, InputInspection, SourceProfile, WorkingImage


RAW_SUFFIXES = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".erf",
    ".fff",
    ".iiq",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".sr2",
    ".srf",
    ".x3f",
}


def is_raw_path(path: Path) -> bool:
    return path.suffix.lower() in RAW_SUFFIXES


def _rawpy():
    try:
        import rawpy  # type: ignore

        return rawpy
    except Exception:
        return None


def inspect_raw(path: Path) -> InputInspection:
    rawpy = _rawpy()
    if rawpy is None:
        return InputInspection(
            path=path,
            exists=path.exists(),
            source_kind="raw",
            format_name=path.suffix.lower().lstrip("."),
            source_profile=SourceProfile("raw_metadata", "LibRaw/rawpy unavailable"),
            warnings=[DecodeWarning("rawpy_unavailable", "rawpy is not installed; RAW inspection is limited.")],
        )
    warnings: list[DecodeWarning] = []
    metadata: dict[str, Any] = {}
    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            metadata = {
                "raw_type": str(raw.raw_type),
                "color_desc": raw.color_desc.decode("ascii", errors="ignore")
                if isinstance(raw.color_desc, bytes)
                else str(raw.color_desc),
                "num_colors": int(raw.num_colors),
                "black_level_per_channel": [int(value) for value in raw.black_level_per_channel],
                "white_level": int(raw.white_level) if raw.white_level is not None else None,
                "camera_whitebalance": [float(value) for value in raw.camera_whitebalance],
                "daylight_whitebalance": [float(value) for value in raw.daylight_whitebalance],
                "raw_width": int(sizes.raw_width),
                "raw_height": int(sizes.raw_height),
                "visible_width": int(sizes.width),
                "visible_height": int(sizes.height),
            }
            return InputInspection(
                path=path,
                exists=True,
                source_kind="raw",
                format_name=path.suffix.lower().lstrip("."),
                width=int(sizes.width),
                height=int(sizes.height),
                bit_depth=16,
                source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata"),
                transfer_state="scene_linear",
                raw_metadata=metadata,
                warnings=warnings,
            )
    except Exception as exc:  # noqa: BLE001 - inspector should return a structured failure.
        return InputInspection(
            path=path,
            exists=path.exists(),
            source_kind="raw",
            format_name=path.suffix.lower().lstrip("."),
            source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata unavailable"),
            raw_metadata=metadata,
            warnings=[DecodeWarning("raw_inspect_failed", repr(exc))],
        )


def load_raw_working_image(path: Path, use_camera_wb: bool = True, no_auto_bright: bool = True) -> WorkingImage:
    rawpy = _rawpy()
    if rawpy is None:
        raise RuntimeError("rawpy is required for RAW decode.")
    inspection = inspect_raw(path)
    warnings = list(inspection.warnings)
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=use_camera_wb,
            no_auto_bright=no_auto_bright,
            output_bps=16,
            gamma=(1, 1),
        )
    pixels = (rgb16.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    warnings.append(
        DecodeWarning(
            "generic_raw_render",
            "RAW decoded through LibRaw/rawpy generic path; exact vendor/Adobe rendering is not promised.",
        )
    )
    return WorkingImage(
        pixels=pixels,
        working_space="camera_rgb_linear",
        transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata"),
        hdr_metadata={},
        orientation_applied=False,
        alpha_policy="absent",
        bit_depth_in=inspection.bit_depth,
        source_path=path,
        warnings=warnings,
    )
