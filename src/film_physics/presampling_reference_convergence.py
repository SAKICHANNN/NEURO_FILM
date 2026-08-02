"""FFT reference utilities for presampling dye-cloud scanner convergence."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .circular_scanner_aperture import compile_circular_aperture_kernel
from .developed_structure import (
    build_colour_dye_cloud_context,
    render_developed_structure,
)


def apply_circular_aperture_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or not np.all(np.isfinite(values))
        or weights.ndim != 2
        or weights.shape[0] != weights.shape[1]
        or weights.shape[0] % 2 != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or abs(float(np.sum(weights)) - 1.0) > 1e-12
    ):
        raise ValueError("invalid FFT circular-aperture request")
    radius = weights.shape[0] // 2
    padded = np.pad(values, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(values)
    for channel in range(3):
        output[..., channel] = fftconvolve(padded[..., channel], weights, mode="valid")
    return output


def _sample_cell_centres(values: np.ndarray, zoom: int) -> np.ndarray:
    image = np.asarray(values, dtype=np.float64)
    if zoom < 2 or zoom % 2 or image.shape[0] % zoom or image.shape[1] % zoom:
        raise ValueError("invalid reference sample geometry")
    blocks = image.reshape(
        image.shape[0] // zoom,
        zoom,
        image.shape[1] // zoom,
        zoom,
        3,
    )
    lower = zoom // 2 - 1
    upper = zoom // 2 + 1
    return blocks[:, lower:upper, :, lower:upper, :].mean(axis=(1, 3), dtype=np.float64)


def render_fft_presampling_reference(
    target_density: np.ndarray,
    *,
    zoom: int,
    target_pixel_pitch_um: float,
    aperture_diameter_um: float,
    aperture_subpixels_per_axis: int,
    radius_um_cmy: tuple[float, float, float],
    mark_optical_density_cmy: tuple[float, float, float],
    monte_carlo_samples: int,
    seed: int,
) -> np.ndarray:
    context = build_colour_dye_cloud_context(
        target_density,
        radius_um_cmy=radius_um_cmy,
        mark_optical_density_cmy=mark_optical_density_cmy,
        output_zoom=zoom,
        output_pixel_pitch_um=target_pixel_pitch_um / zoom,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    density = np.asarray(render_developed_structure(context).values, dtype=np.float64)
    transmittance = np.power(10.0, -density)
    kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=aperture_diameter_um,
        pixel_pitch_um=target_pixel_pitch_um / zoom,
        subpixels_per_axis=aperture_subpixels_per_axis,
    )
    scanned = apply_circular_aperture_fft(transmittance, kernel)
    output = _sample_cell_centres(scanned, zoom)
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0) or np.any(output > 1.0):
        raise RuntimeError("FFT presampling reference left transmittance domain")
    return output


__all__ = ["apply_circular_aperture_fft", "render_fft_presampling_reference"]
