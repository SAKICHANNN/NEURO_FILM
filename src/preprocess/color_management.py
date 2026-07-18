"""Explicit dependency-free linear RGB working-space conversions."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .types import DecodeWarning, WorkingImage


LINEAR_RGB_TRANSFORM_VERSION = "linear-d65-srgb-rec2020-v1"
REC2020_TRANSFER_VERSION = "bt2020-2-oetf-v1"
REC2020_SDR_CICP = bytes((9, 15, 0, 1))
_SUPPORTED_SPACES = frozenset({"linear_srgb", "linear_rec2020"})

_BT2020_ALPHA = 1.09929682680944
_BT2020_BETA = 0.018053968510807

# W3C CSS Color 4 rational matrices, column-vector convention, D65 XYZ.
_LINEAR_SRGB_TO_XYZ_D65 = np.asarray(
    [
        [506752 / 1228815, 87881 / 245763, 12673 / 70218],
        [87098 / 409605, 175762 / 245763, 12673 / 175545],
        [7918 / 409605, 87881 / 737289, 1001167 / 1053270],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_SRGB = np.asarray(
    [
        [12831 / 3959, -329 / 214, -1974 / 3959],
        [-851781 / 878810, 1648619 / 878810, 36519 / 878810],
        [705 / 12673, -2585 / 12673, 705 / 667],
    ],
    dtype=np.float64,
)
_LINEAR_REC2020_TO_XYZ_D65 = np.asarray(
    [
        [63426534 / 99577255, 20160776 / 139408157, 47086771 / 278816314],
        [26158966 / 99577255, 472592308 / 697040785, 8267143 / 139408157],
        [0.0, 19567812 / 697040785, 295819943 / 278816314],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_REC2020 = np.asarray(
    [
        [30757411 / 17917100, -6372589 / 17917100, -4539589 / 17917100],
        [-19765991 / 29648200, 47925759 / 29648200, 467509 / 29648200],
        [792561 / 44930125, -1921689 / 44930125, 42328811 / 44930125],
    ],
    dtype=np.float64,
)

_MATRICES = {
    ("linear_srgb", "linear_rec2020"): (
        _XYZ_D65_TO_LINEAR_REC2020 @ _LINEAR_SRGB_TO_XYZ_D65
    ),
    ("linear_rec2020", "linear_srgb"): (
        _XYZ_D65_TO_LINEAR_SRGB @ _LINEAR_REC2020_TO_XYZ_D65
    ),
}


def _space(value: object, label: str) -> str:
    if not isinstance(value, str) or value not in _SUPPORTED_SPACES:
        allowed = ", ".join(sorted(_SUPPORTED_SPACES))
        raise ValueError(f"{label} must be one of: {allowed}")
    return value


def linear_rgb_matrix(source_space: str, destination_space: str) -> np.ndarray:
    """Return an independent float64 column-vector conversion matrix."""
    source = _space(source_space, "source_space")
    destination = _space(destination_space, "destination_space")
    if source == destination:
        return np.eye(3, dtype=np.float64)
    return _MATRICES[(source, destination)].copy()


def convert_linear_rgb(
    pixels: np.ndarray,
    *,
    source_space: str,
    destination_space: str,
) -> np.ndarray:
    """Convert finite float32 HxWx3 linear RGB without clipping or tone mapping."""
    if not isinstance(pixels, np.ndarray):
        raise TypeError("pixels must be a numpy ndarray")
    if pixels.dtype != np.float32:
        raise TypeError("pixels must be float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must be HxWx3")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must contain only finite values")
    matrix = linear_rgb_matrix(source_space, destination_space)
    if source_space == destination_space:
        return pixels.copy()
    converted = np.matmul(pixels.astype(np.float64), matrix.T).astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("working-space conversion produced non-finite values")
    return converted


def _finite_float32_rgb(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray):
        raise TypeError("pixels must be a numpy ndarray")
    if pixels.dtype != np.float32:
        raise TypeError("pixels must be float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must be HxWx3")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must contain only finite values")


def linear_rec2020_to_rec2020(pixels: np.ndarray) -> np.ndarray:
    """Apply the signed BT.2020-2 opto-electronic transfer function."""
    _finite_float32_rgb(pixels)
    absolute = np.abs(pixels.astype(np.float64))
    encoded = np.where(
        absolute < _BT2020_BETA,
        4.5 * absolute,
        _BT2020_ALPHA * np.power(absolute, 0.45) - (_BT2020_ALPHA - 1.0),
    )
    return np.asarray(np.copysign(encoded, pixels), dtype=np.float32)


def rec2020_to_linear_rec2020(pixels: np.ndarray) -> np.ndarray:
    """Invert the signed BT.2020-2 opto-electronic transfer function."""
    _finite_float32_rgb(pixels)
    absolute = np.abs(pixels.astype(np.float64))
    linear = np.where(
        absolute < 4.5 * _BT2020_BETA,
        absolute / 4.5,
        np.power(
            (absolute + (_BT2020_ALPHA - 1.0)) / _BT2020_ALPHA,
            1.0 / 0.45,
        ),
    )
    return np.asarray(np.copysign(linear, pixels), dtype=np.float32)


def convert_working_image_space(
    working: WorkingImage, destination_space: str
) -> WorkingImage:
    """Return a provenance-preserving WorkingImage in another linear D65 space."""
    if not isinstance(working, WorkingImage):
        raise TypeError("working must be a WorkingImage")
    if working.transfer_state not in {"scene_linear", "display_linear"}:
        raise ValueError("working-space conversion requires linear transfer_state")
    destination = _space(destination_space, "destination_space")
    source = _space(working.working_space, "working.working_space")
    converted = convert_linear_rgb(
        working.pixels,
        source_space=source,
        destination_space=destination,
    )
    warnings = list(working.warnings)
    if source != destination:
        warnings.append(
            DecodeWarning(
                "working_space_conversion",
                f"Converted {source} to {destination} with "
                f"{LINEAR_RGB_TRANSFORM_VERSION}; no clipping or gamut mapping.",
            )
        )
    return WorkingImage(
        pixels=converted,
        working_space=destination,
        transfer_state=working.transfer_state,
        source_transfer_state=working.source_transfer_state,
        source_profile=working.source_profile,
        hdr_metadata=deepcopy(working.hdr_metadata),
        orientation_applied=working.orientation_applied,
        alpha_policy=working.alpha_policy,
        bit_depth_in=working.bit_depth_in,
        source_path=working.source_path,
        warnings=warnings,
    )
