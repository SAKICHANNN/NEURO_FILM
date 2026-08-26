"""Private exact-source DNG to official ACES 2 XYZ-D65-nits bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dng_forward_raster import load_dng_forward_working_image
from .ocio_aces2_output import (
    OcioAces2RuntimeError,
    _ocio,
    convert_working_image_to_acescg,
)
from .types import WorkingImage

SCENE_TO_REFERENCE_BUILTIN = "ACEScg_to_ACES2065-1"
OUTPUT_BUILTIN = "ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-1000nit-P3-D65_2.0"
DISPLAY_REFERENCE_SCALE_NITS = np.float32(100.0)
OUTPUT_PROFILE_ID = "display-absolute-linear-xyz-d65.v1"


def render_acescg_to_xyz_d65_nits(values: np.ndarray) -> np.ndarray:
    """Render finite float32 ACEScg values to owned absolute XYZ-D65."""

    source = np.asarray(values)
    if (
        source.dtype != np.float32
        or source.ndim < 2
        or source.shape[-1] != 3
        or source.size == 0
        or not np.isfinite(source).all()
    ):
        raise OcioAces2RuntimeError(
            "ACEScg XYZ bridge requires non-empty finite float32 RGB values"
        )
    output = np.ascontiguousarray(source.copy())
    flat = output.reshape(-1, 3)
    ocio = _ocio()
    group = ocio.GroupTransform()
    group.appendTransform(ocio.BuiltinTransform(SCENE_TO_REFERENCE_BUILTIN))
    group.appendTransform(ocio.BuiltinTransform(OUTPUT_BUILTIN))
    processor = ocio.Config.CreateRaw().getProcessor(group).getDefaultCPUProcessor()
    processor.apply(ocio.PackedImageDesc(flat, flat.shape[0], 1, 3))
    output *= DISPLAY_REFERENCE_SCALE_NITS
    if not np.isfinite(output).all():
        raise OcioAces2RuntimeError("official ACES 2 XYZ output is non-finite")
    return output


def render_working_image_to_xyz_d65_nits(working: WorkingImage) -> np.ndarray:
    """Render a supported scene-linear WorkingImage to absolute XYZ-D65."""

    acescg = convert_working_image_to_acescg(working)
    return render_acescg_to_xyz_d65_nits(acescg)


def load_dng_forward_aces2_xyz_d65_nits(
    path: Path | str,
    *,
    expected_source_bytes: int,
    expected_source_sha256: str,
) -> np.ndarray:
    """Load one exact DNG through P98 and return owned XYZ-D65 nits."""

    working = load_dng_forward_working_image(
        path,
        expected_source_bytes=expected_source_bytes,
        expected_source_sha256=expected_source_sha256,
    )
    return render_working_image_to_xyz_d65_nits(working)


__all__ = [
    "DISPLAY_REFERENCE_SCALE_NITS",
    "OUTPUT_BUILTIN",
    "OUTPUT_PROFILE_ID",
    "SCENE_TO_REFERENCE_BUILTIN",
    "load_dng_forward_aces2_xyz_d65_nits",
    "render_acescg_to_xyz_d65_nits",
    "render_working_image_to_xyz_d65_nits",
]
