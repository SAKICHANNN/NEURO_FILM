"""Constant-preserving positive-lobe backing-return approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .backing_return import BackingReturnProfile, backing_return_kernel_2d
from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit


@dataclass(frozen=True)
class PositiveSpreadReturn:
    exposure: PhysicalDomainArray
    residual: np.ndarray


def _symmetric_fft_convolve2d(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT convolution with the same symmetric-context domain as the reference."""

    radius_y = kernel.shape[0] // 2
    radius_x = kernel.shape[1] // 2
    padded = np.pad(
        values,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode="symmetric",
    )
    convolved = fftconvolve(padded, kernel, mode="same")
    return convolved[
        radius_y : radius_y + values.shape[0],
        radius_x : radius_x + values.shape[1],
    ]


def apply_positive_spread_backing_return(
    exposure: PhysicalDomainArray,
    profile: BackingReturnProfile,
) -> PositiveSpreadReturn:
    """Add only the positive exterior lobe of each fixed return blur.

    This is a compiled, nonlinear approximation for visible highlight spread,
    not the linear transport reference. Symmetric context makes constants exact
    and avoids treating a crop boundary as the physical film edge.
    """

    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    values = exposure.values
    if values.dtype != np.float64:
        raise TypeError("positive spread requires float64 exposure")
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positive spread requires HxWx3 exposure")
    if exposure.scale != profile.scale:
        raise ValueError("positive-spread pixel scale does not match profile")
    residual = np.zeros_like(values)
    tolerance = 64.0 * np.finfo(np.float64).eps
    for component in profile.components:
        kernel = backing_return_kernel_2d(component, profile.scale)
        weights = component.return_weights
        for source_channel in range(3):
            blurred = _symmetric_fft_convolve2d(
                values[..., source_channel], kernel
            )
            positive = blurred - values[..., source_channel]
            positive = np.where(positive > tolerance, positive, 0.0)
            for target_layer in range(3):
                weight = float(weights[target_layer, source_channel])
                if weight:
                    residual[..., target_layer] += weight * positive
    output = values + residual
    if not np.all(np.isfinite(output)) or np.any(output < values):
        raise RuntimeError("positive spread left its additive exposure domain")
    return PositiveSpreadReturn(
        PhysicalDomainArray(
            output,
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            exposure.channels,
            exposure.scale,
        ),
        residual,
    )


__all__ = [
    "PositiveSpreadReturn",
    "_symmetric_fft_convolve2d",
    "apply_positive_spread_backing_return",
]
