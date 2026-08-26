"""Strict opt-in ACES2065-1 OpenEXR to ACEScg WorkingImage ingress."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .types import SourceProfile, WorkingImage

ACES2065_OPENEXR_INGRESS_ID = "aces2065-ap0-openexr-to-acescg-working-image-v1"
ACES2065_COLOR_INTEROP_ID = "lin_ap0_scene"
ACES2065_MAXIMUM_WIDTH = 16_384
ACES2065_MAXIMUM_HEIGHT = 16_384
ACES2065_MAXIMUM_PIXELS = 100_000_000
ACES2065_MAXIMUM_ABSOLUTE_COMPONENT = 65_504.0

ACES2065_CHROMATICITIES = (
    0.7347000241279602,
    0.2653000056743622,
    0.0,
    1.0,
    0.00009999999747378752,
    -0.07699999958276749,
    0.3216800093650818,
    0.3376699984073639,
)
ACES2065_ADOPTED_NEUTRAL = (
    0.3216800093650818,
    0.3376699984073639,
)
ACES2065_TO_ACESCG_MATRIX = np.asarray(
    [
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734, 1.1762296998, -0.0996759264],
        [0.0083161484, -0.0060324498, 0.9977163014],
    ],
    dtype=np.float64,
)


class Aces2065OpenExrError(ValueError):
    """Raised when an EXR violates the strict P251 ACES2065-1 contract."""


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _float_tuple(value: object, *, label: str, length: int) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise Aces2065OpenExrError(f"invalid {label}") from exc
    if len(result) != length or not np.isfinite(result).all():
        raise Aces2065OpenExrError(f"invalid {label}")
    return result


def _validate_header(header: dict[str, Any], openexr: ModuleType) -> None:
    required = {
        "acesImageContainerFlag",
        "adoptedNeutral",
        "chromaticities",
        "colorInteropID",
        "type",
    }
    if not required.issubset(header):
        raise Aces2065OpenExrError("missing required ACES2065-1 metadata")
    if header["type"] != openexr.scanlineimage:
        raise Aces2065OpenExrError("only non-deep scanline OpenEXR is accepted")
    if int(header["acesImageContainerFlag"]) != 1:
        raise Aces2065OpenExrError("acesImageContainerFlag must equal 1")
    if _text(header["colorInteropID"]) != ACES2065_COLOR_INTEROP_ID:
        raise Aces2065OpenExrError("colorInteropID must identify lin_ap0_scene")
    if _float_tuple(
        header["chromaticities"], label="chromaticities", length=8
    ) != ACES2065_CHROMATICITIES:
        raise Aces2065OpenExrError("chromaticities do not match AP0/D60")
    if _float_tuple(
        header["adoptedNeutral"], label="adoptedNeutral", length=2
    ) != ACES2065_ADOPTED_NEUTRAL:
        raise Aces2065OpenExrError("adoptedNeutral does not match D60")


def _validate_pixels(pixels: object) -> np.ndarray:
    if not isinstance(pixels, np.ndarray):
        raise Aces2065OpenExrError("packed RGB pixels must be a numpy array")
    if pixels.dtype != np.float32:
        raise Aces2065OpenExrError("packed RGB pixels must be float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise Aces2065OpenExrError("packed RGB pixels must be HxWx3")
    height, width, _ = pixels.shape
    if height <= 0 or width <= 0:
        raise Aces2065OpenExrError("OpenEXR dimensions must be positive")
    if height > ACES2065_MAXIMUM_HEIGHT or width > ACES2065_MAXIMUM_WIDTH:
        raise Aces2065OpenExrError("OpenEXR dimensions exceed the frozen ceiling")
    if height * width > ACES2065_MAXIMUM_PIXELS:
        raise Aces2065OpenExrError("OpenEXR pixel count exceeds the frozen ceiling")
    if not np.isfinite(pixels).all():
        raise Aces2065OpenExrError("OpenEXR pixels must be finite")
    if float(np.max(np.abs(pixels))) > ACES2065_MAXIMUM_ABSOLUTE_COMPONENT:
        raise Aces2065OpenExrError("OpenEXR component exceeds the frozen ceiling")
    return pixels


def _load_with_module(path: Path, openexr: ModuleType) -> WorkingImage:
    try:
        with openexr.File(str(path)) as exr_file:
            parts = getattr(exr_file, "parts", None)
            if parts is None or len(parts) != 1:
                raise Aces2065OpenExrError("exactly one OpenEXR part is required")
            header = dict(exr_file.header())
            channels = exr_file.channels()
            _validate_header(header, openexr)
            if sorted(channels) != ["RGB"]:
                raise Aces2065OpenExrError(
                    "exactly one packed RGB channel group is required"
                )
            ap0 = _validate_pixels(channels["RGB"].pixels)
            acescg = np.ascontiguousarray(
                np.asarray(
                    ap0.astype(np.float64) @ ACES2065_TO_ACESCG_MATRIX.T,
                    dtype=np.float32,
                )
            )
    except Aces2065OpenExrError:
        raise
    except Exception as exc:
        raise Aces2065OpenExrError("failed to read strict ACES2065-1 OpenEXR") from exc
    if not np.isfinite(acescg).all():
        raise Aces2065OpenExrError("AP0 to AP1 conversion produced non-finite values")
    return WorkingImage(
        pixels=acescg.copy(),
        working_space="acescg_ap1_d60",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile(
            "icc", "ACES2065-1 AP0/D60 OpenEXR", bytes_length=0
        ),
        hdr_metadata={
            "aces_image_container_flag": 1,
            "adopted_neutral": list(ACES2065_ADOPTED_NEUTRAL),
            "chromaticities": list(ACES2065_CHROMATICITIES),
            "color_interop_id": ACES2065_COLOR_INTEROP_ID,
            "ingress_id": ACES2065_OPENEXR_INGRESS_ID,
        },
        orientation_applied=False,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=path,
    )


def load_aces2065_openexr_working_image(path: Path | str) -> WorkingImage:
    """Load one strictly identified AP0/D60 ACES2065-1 EXR into ACEScg/AP1."""
    source = Path(path)
    if not source.is_file():
        raise Aces2065OpenExrError("source must be an existing file")
    try:
        openexr = importlib.import_module("OpenEXR")
    except ImportError as exc:
        raise Aces2065OpenExrError(
            "OpenEXR 3.4.15 is required for the opt-in ACES2065-1 ingress"
        ) from exc
    if getattr(openexr, "__version__", None) != "3.4.15":
        raise Aces2065OpenExrError("OpenEXR runtime must be exactly 3.4.15")
    return _load_with_module(source, openexr)


__all__ = [
    "ACES2065_OPENEXR_INGRESS_ID",
    "Aces2065OpenExrError",
    "load_aces2065_openexr_working_image",
]
