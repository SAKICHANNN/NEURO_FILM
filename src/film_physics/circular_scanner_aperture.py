"""Positive reference compiler for a circular scanner sampling aperture."""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import j1


class CircularApertureError(ValueError):
    """Raised when a circular-aperture request violates the reference contract."""


def analytic_circular_aperture_mtf(
    frequencies_cycles_per_mm: np.ndarray,
    aperture_diameter_um: float,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0.0)
        or not math.isfinite(aperture_diameter_um)
        or aperture_diameter_um <= 0.0
    ):
        raise CircularApertureError("invalid circular-aperture MTF request")
    argument = math.pi * (aperture_diameter_um / 1000.0) * frequencies
    response = np.ones_like(argument)
    nonzero = argument != 0.0
    response[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
    return response


def compile_circular_aperture_kernel(
    *,
    aperture_diameter_um: float,
    pixel_pitch_um: float,
    subpixels_per_axis: int,
) -> np.ndarray:
    if (
        not math.isfinite(aperture_diameter_um)
        or aperture_diameter_um <= 0.0
        or not math.isfinite(pixel_pitch_um)
        or pixel_pitch_um <= 0.0
        or not isinstance(subpixels_per_axis, int)
        or isinstance(subpixels_per_axis, bool)
        or subpixels_per_axis < 2
    ):
        raise CircularApertureError("invalid circular-aperture compiler request")
    radius_um = aperture_diameter_um / 2.0
    half = math.ceil(radius_um / pixel_pitch_um + 0.5)
    offsets = (
        (np.arange(subpixels_per_axis, dtype=np.float64) + 0.5) / subpixels_per_axis
        - 0.5
    ) * pixel_pitch_um
    coordinates = np.arange(-half, half + 1, dtype=np.float64) * pixel_pitch_um
    kernel = np.zeros((coordinates.size, coordinates.size), dtype=np.float64)
    for row, y in enumerate(coordinates):
        y_squared = np.square(y + offsets)[:, None]
        for column, x in enumerate(coordinates):
            inside = y_squared + np.square(x + offsets)[None, :] <= radius_um**2
            kernel[row, column] = float(np.mean(inside))
    positive_rows = np.any(kernel > 0.0, axis=1)
    positive_columns = np.any(kernel > 0.0, axis=0)
    if not np.any(positive_rows) or not np.any(positive_columns):
        raise CircularApertureError("compiled circular aperture is empty")
    kernel = kernel[np.ix_(positive_rows, positive_columns)]
    if kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise CircularApertureError("compiled circular aperture support drift")
    total = float(np.sum(kernel, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise CircularApertureError("compiled circular aperture normalization failed")
    kernel /= total
    return kernel


def sampled_kernel_mtf(
    kernel: np.ndarray,
    frequencies_cycles_per_mm: np.ndarray,
    pixel_pitch_um: float,
) -> np.ndarray:
    weights = np.asarray(kernel, dtype=np.float64)
    frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
    if (
        weights.ndim != 2
        or weights.shape[0] != weights.shape[1]
        or weights.shape[0] % 2 != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or frequencies.ndim != 1
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0.0)
        or not math.isfinite(pixel_pitch_um)
        or pixel_pitch_um <= 0.0
    ):
        raise CircularApertureError("invalid sampled-kernel MTF request")
    half = weights.shape[1] // 2
    x_mm = np.arange(-half, half + 1, dtype=np.float64) * pixel_pitch_um / 1000.0
    marginal = np.sum(weights, axis=0, dtype=np.float64)
    return np.asarray(
        [
            abs(np.sum(marginal * np.exp(-2j * math.pi * frequency * x_mm)))
            for frequency in frequencies
        ],
        dtype=np.float64,
    )


def _convolve_padded_rows(
    padded: np.ndarray,
    kernel: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    width: int,
) -> np.ndarray:
    radius = kernel.shape[0] // 2
    segment = padded[row_start : row_stop + 2 * radius]
    windows = sliding_window_view(
        segment,
        (kernel.shape[0], kernel.shape[1]),
        axis=(0, 1),
    )
    if windows.shape[:2] != (row_stop - row_start, width):
        raise CircularApertureError("circular-aperture window geometry drift")
    return np.einsum("hwcij,ij->hwc", windows, kernel, optimize=False)


def apply_circular_aperture(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    row_partition: int | None = None,
) -> np.ndarray:
    values = np.asarray(image)
    weights = np.asarray(kernel, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or values.dtype != np.float64
        or not np.all(np.isfinite(values))
        or weights.ndim != 2
        or weights.shape[0] != weights.shape[1]
        or weights.shape[0] % 2 != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
    ):
        raise CircularApertureError("invalid circular-aperture image request")
    if row_partition is not None and (
        not isinstance(row_partition, int)
        or isinstance(row_partition, bool)
        or row_partition <= 0
    ):
        raise CircularApertureError("invalid circular-aperture row partition")
    radius = weights.shape[0] // 2
    padded = np.pad(values, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(values)
    step = values.shape[0] if row_partition is None else row_partition
    for row_start in range(0, values.shape[0], step):
        row_stop = min(row_start + step, values.shape[0])
        output[row_start:row_stop] = _convolve_padded_rows(
            padded,
            weights,
            row_start=row_start,
            row_stop=row_stop,
            width=values.shape[1],
        )
    return output


__all__ = [
    "CircularApertureError",
    "analytic_circular_aperture_mtf",
    "apply_circular_aperture",
    "compile_circular_aperture_kernel",
    "sampled_kernel_mtf",
]
