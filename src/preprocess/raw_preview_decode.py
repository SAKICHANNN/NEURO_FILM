"""Private native half-size RAW decode for bounded product previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .raw_decode import _rawpy, inspect_raw
from .types import DecodeWarning, SourceProfile, WorkingImage


def load_raw_preview_working_image(
    path: Path,
    *,
    use_camera_wb: bool = True,
    no_auto_bright: bool = True,
) -> WorkingImage:
    """Decode one RAW through LibRaw's native half-size demosaic.

    This is an explicit preview-only entry point. The general RAW ingress keeps
    its historical full-resolution behavior.
    """

    rawpy = _rawpy()
    if rawpy is None:
        raise RuntimeError("rawpy is required for RAW preview decode.")
    inspection = inspect_raw(path)
    if not inspection.exists or inspection.width < 1 or inspection.height < 1:
        raise ValueError("RAW preview decode requires a valid inspectable RAW file")
    warnings = list(inspection.warnings)
    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=use_camera_wb,
            no_auto_bright=no_auto_bright,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(1, 1),
            user_flip=None,
            half_size=True,
        )
    pixels = np.ascontiguousarray((rgb16.astype(np.float32) / 65535.0).clip(0.0, 1.0))
    warnings.extend(
        [
            DecodeWarning(
                "generic_raw_render",
                "RAW decoded through LibRaw/rawpy generic path; exact vendor/Adobe rendering is not promised.",
            ),
            DecodeWarning(
                "generic_raw_display_mapping",
                "Linear-sRGB RAW decode enters the SDR sRGB look pipeline without a calibrated scene-to-display tone map.",
            ),
            DecodeWarning(
                "raw_half_size_preview_decode",
                "RAW preview uses LibRaw native half-size demosaic before float expansion; final export remains full-resolution.",
            ),
        ]
    )
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="scene_linear",
        source_transfer_state=inspection.transfer_state,
        source_profile=SourceProfile("raw_metadata", "LibRaw camera metadata"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=inspection.bit_depth,
        source_path=path,
        warnings=warnings,
    )


__all__ = ["load_raw_preview_working_image"]
