"""Explicit immutable-source translation sampling for gate weave."""

from __future__ import annotations

from typing import Literal

import numpy as np


class GateWeaveSamplingError(ValueError):
    """Raised when a translation cannot be sampled from the declared padding."""


def sample_padded_translation(
    padded_image: np.ndarray,
    *,
    output_shape: tuple[int, int],
    padding_yx: tuple[int, int],
    offset_yx: tuple[float, float],
    interpolation: Literal["nearest", "bilinear"],
) -> np.ndarray:
    """Sample one translated output directly from an immutable padded source."""
    source = np.asarray(padded_image)
    if source.ndim not in {2, 3} or not np.issubdtype(source.dtype, np.floating):
        raise GateWeaveSamplingError("source must be a floating 2-D or HWC array")
    height, width = output_shape
    padding_y, padding_x = padding_yx
    if height < 1 or width < 1 or padding_y < 0 or padding_x < 0:
        raise GateWeaveSamplingError("invalid output shape or padding")
    if source.shape[:2] != (height + 2 * padding_y, width + 2 * padding_x):
        raise GateWeaveSamplingError("source shape does not match output and padding")
    offset_y, offset_x = offset_yx
    if not np.isfinite(offset_y) or not np.isfinite(offset_x):
        raise GateWeaveSamplingError("translation offset must be finite")
    if abs(offset_y) > padding_y or abs(offset_x) > padding_x:
        raise GateWeaveSamplingError("translation exceeds declared padding")

    if interpolation == "nearest":
        integer_y = int(np.rint(offset_y))
        integer_x = int(np.rint(offset_x))
        y0 = padding_y + integer_y
        x0 = padding_x + integer_x
        return source[y0 : y0 + height, x0 : x0 + width].copy()
    if interpolation != "bilinear":
        raise GateWeaveSamplingError("unsupported interpolation")

    sample_y = padding_y + np.arange(height, dtype=np.float64) + offset_y
    sample_x = padding_x + np.arange(width, dtype=np.float64) + offset_x
    y0 = np.floor(sample_y).astype(np.int64)
    x0 = np.floor(sample_x).astype(np.int64)
    fy = sample_y - y0
    fx = sample_x - x0
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    if (
        y0.min() < 0
        or x0.min() < 0
        or y0.max() >= source.shape[0]
        or x0.max() >= source.shape[1]
    ):
        raise GateWeaveSamplingError("translation samples outside the padded source")
    if source.ndim == 3:
        fy = fy[:, None, None]
        fx = fx[None, :, None]
    else:
        fy = fy[:, None]
        fx = fx[None, :]
    top = (
        source[y0[:, None], x0[None, :]] * (1.0 - fx)
        + source[y0[:, None], x1[None, :]] * fx
    )
    bottom = (
        source[y1[:, None], x0[None, :]] * (1.0 - fx)
        + source[y1[:, None], x1[None, :]] * fx
    )
    return top * (1.0 - fy) + bottom * fy
