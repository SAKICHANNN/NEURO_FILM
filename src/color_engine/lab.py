"""Explicit D65 CIELAB conversions for supported linear working spaces."""

from __future__ import annotations

import numpy as np

from src.preprocess.color_management import linear_rgb_matrix


_D65_WHITE = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)

# The current safe-Lab baseline uses these skimage matrices and its historical
# EasyRGB Lab constants. Rec.2020 is mapped into that same legacy PCS so a
# physical colour has one Lab value across both working spaces.
_LEGACY_SRGB_TO_XYZ = np.asarray(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float32,
)
_LEGACY_XYZ_TO_SRGB = np.asarray(
    [
        [3.24048134, -1.53715152, -0.49853633],
        [-0.96925495, 1.87599000, 0.04155593],
        [0.05564664, -0.20404134, 1.05731107],
    ],
    dtype=np.float32,
)


def _matrices(working_space: str) -> tuple[np.ndarray, np.ndarray]:
    if working_space == "linear_srgb":
        return _LEGACY_SRGB_TO_XYZ, _LEGACY_XYZ_TO_SRGB
    if working_space == "linear_rec2020":
        to_srgb = linear_rgb_matrix("linear_rec2020", "linear_srgb")
        from_srgb = linear_rgb_matrix("linear_srgb", "linear_rec2020")
        return (
            np.asarray(_LEGACY_SRGB_TO_XYZ @ to_srgb, dtype=np.float64),
            np.asarray(from_srgb @ _LEGACY_XYZ_TO_SRGB, dtype=np.float64),
        )
    raise ValueError("working_space must be linear_srgb or linear_rec2020")


def _finite_float32_three_channel(value: np.ndarray, label: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{label} must be float32")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{label} must be HxWx3")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must contain only finite values")


def linear_rgb_to_lab(pixels: np.ndarray, *, working_space: str) -> np.ndarray:
    """Convert supported linear RGB to D65 CIELAB without gamut clipping."""
    _finite_float32_three_channel(pixels, "pixels")
    to_xyz, _ = _matrices(working_space)
    xyz = np.matmul(pixels, to_xyz.T).astype(np.float32)
    relative = xyz / _D65_WHITE
    f = np.where(
        relative > 0.008856,
        np.cbrt(relative),
        7.787 * relative + 16.0 / 116.0,
    )
    lab = np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )
    return np.asarray(lab, dtype=np.float32)


def lab_to_linear_rgb(lab: np.ndarray, *, working_space: str) -> np.ndarray:
    """Convert D65 CIELAB to supported linear RGB without gamut clipping."""
    _finite_float32_three_channel(lab, "lab")
    value = lab.astype(np.float64)
    fy = (value[..., 0] + 16.0) / 116.0
    fx = fy + value[..., 1] / 500.0
    fz = fy - value[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    relative = np.where(
        f > 0.2068966,
        f**3,
        (f - 16.0 / 116.0) / 7.787,
    )
    xyz = np.asarray(relative * _D65_WHITE, dtype=np.float32)
    _, from_xyz = _matrices(working_space)
    return np.asarray(np.matmul(xyz, from_xyz.T), dtype=np.float32)
