"""Validated deterministic IEC sRGB transfer functions."""

from __future__ import annotations

import numpy as np


class SrgbTransferError(ValueError):
    """Raised when an sRGB transfer input violates the RGB contract."""


def encoded_srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
    ):
        raise SrgbTransferError("encoded sRGB must be finite HxWx3")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise SrgbTransferError("encoded sRGB must be in [0, 1]")
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def linear_srgb_to_encoded(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
    ):
        raise SrgbTransferError("linear sRGB must be finite HxWx3")
    if np.any(value < -1e-12) or np.any(value > 1.0 + 1e-12):
        raise SrgbTransferError("linear sRGB escaped [0, 1]")
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


__all__ = [
    "SrgbTransferError",
    "encoded_srgb_to_linear",
    "linear_srgb_to_encoded",
]
