"""Offline dye-density to transmittance presampling scanner reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .circular_scanner_aperture import (
    apply_circular_aperture,
    compile_circular_aperture_kernel,
)
from .developed_structure import (
    build_colour_dye_cloud_context,
    render_developed_structure,
)


@dataclass(frozen=True)
class PresamplingDyeCloudScanResult:
    reference: np.ndarray
    candidate: np.ndarray
    wrong_order: np.ndarray
    post_raster_circular: np.ndarray
    post_raster_gaussian: np.ndarray
    unfiltered_target_samples: np.ndarray
    candidate_kernel: np.ndarray
    reference_kernel: np.ndarray


def _sample_exact_cell_centres(values: np.ndarray, factor: int) -> np.ndarray:
    image = np.asarray(values, dtype=np.float64)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or not isinstance(factor, int)
        or isinstance(factor, bool)
        or factor < 2
        or factor % 2 != 0
        or image.shape[0] % factor
        or image.shape[1] % factor
        or not np.all(np.isfinite(image))
    ):
        raise ValueError("invalid exact-cell-centre sampling request")
    blocks = image.reshape(
        image.shape[0] // factor,
        factor,
        image.shape[1] // factor,
        factor,
        3,
    )
    lower = factor // 2 - 1
    upper = factor // 2 + 1
    output = blocks[:, lower:upper, :, lower:upper, :].mean(
        axis=(1, 3), dtype=np.float64
    )
    return np.asarray(output, dtype=np.float64)


def _render_density(
    target_density: np.ndarray,
    *,
    zoom: int,
    target_pixel_pitch_um: float,
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
    return np.asarray(render_developed_structure(context).values, dtype=np.float64)


def _scan_transmittance(
    density: np.ndarray,
    *,
    zoom: int,
    target_pixel_pitch_um: float,
    aperture_diameter_um: float,
    aperture_subpixels_per_axis: int,
    row_partition: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    transmittance = np.power(10.0, -np.asarray(density, dtype=np.float64))
    kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=aperture_diameter_um,
        pixel_pitch_um=target_pixel_pitch_um / zoom,
        subpixels_per_axis=aperture_subpixels_per_axis,
    )
    scanned = apply_circular_aperture(
        transmittance,
        kernel,
        row_partition=None if row_partition is None else row_partition * zoom,
    )
    return _sample_exact_cell_centres(scanned, zoom), kernel


def render_presampling_dye_cloud_scan(
    target_density: np.ndarray,
    *,
    target_pixel_pitch_um: float,
    candidate_zoom: int,
    reference_zoom: int,
    aperture_diameter_um: float,
    aperture_subpixels_per_axis: int,
    radius_um_cmy: tuple[float, float, float],
    mark_optical_density_cmy: tuple[float, float, float],
    monte_carlo_samples: int,
    seed: int,
    row_partition: int | None = None,
) -> PresamplingDyeCloudScanResult:
    """Render the physical ordering and fixed post-raster controls."""

    target = np.asarray(target_density, dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[2] != 3
        or not target.size
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or not math.isfinite(target_pixel_pitch_um)
        or target_pixel_pitch_um <= 0.0
        or not isinstance(candidate_zoom, int)
        or not isinstance(reference_zoom, int)
        or candidate_zoom < 2
        or reference_zoom <= candidate_zoom
        or candidate_zoom % 2
        or reference_zoom % 2
    ):
        raise ValueError("invalid presampling dye-cloud scan request")

    candidate_density = _render_density(
        target,
        zoom=candidate_zoom,
        target_pixel_pitch_um=target_pixel_pitch_um,
        radius_um_cmy=radius_um_cmy,
        mark_optical_density_cmy=mark_optical_density_cmy,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    reference_density = _render_density(
        target,
        zoom=reference_zoom,
        target_pixel_pitch_um=target_pixel_pitch_um,
        radius_um_cmy=radius_um_cmy,
        mark_optical_density_cmy=mark_optical_density_cmy,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    candidate, candidate_kernel = _scan_transmittance(
        candidate_density,
        zoom=candidate_zoom,
        target_pixel_pitch_um=target_pixel_pitch_um,
        aperture_diameter_um=aperture_diameter_um,
        aperture_subpixels_per_axis=aperture_subpixels_per_axis,
        row_partition=row_partition,
    )
    reference, reference_kernel = _scan_transmittance(
        reference_density,
        zoom=reference_zoom,
        target_pixel_pitch_um=target_pixel_pitch_um,
        aperture_diameter_um=aperture_diameter_um,
        aperture_subpixels_per_axis=aperture_subpixels_per_axis,
    )

    candidate_density_blurred = apply_circular_aperture(
        candidate_density,
        candidate_kernel,
        row_partition=None if row_partition is None else row_partition * candidate_zoom,
    )
    wrong_order = np.power(
        10.0,
        -_sample_exact_cell_centres(candidate_density_blurred, candidate_zoom),
    )
    unfiltered = _sample_exact_cell_centres(
        np.power(10.0, -candidate_density), candidate_zoom
    )
    post_kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=aperture_diameter_um,
        pixel_pitch_um=target_pixel_pitch_um,
        subpixels_per_axis=aperture_subpixels_per_axis,
    )
    post_circular = apply_circular_aperture(
        unfiltered, post_kernel, row_partition=row_partition
    )
    sigma_pixels = (aperture_diameter_um / 4.0) / target_pixel_pitch_um
    post_gaussian = gaussian_filter(
        unfiltered,
        sigma=(sigma_pixels, sigma_pixels, 0.0),
        order=0,
        mode="nearest",
        truncate=4.0,
    )
    outputs = (
        reference,
        candidate,
        wrong_order,
        post_circular,
        post_gaussian,
        unfiltered,
    )
    if any(
        not np.all(np.isfinite(output)) or np.any(output <= 0.0) or np.any(output > 1.0)
        for output in outputs
    ):
        raise RuntimeError("presampling dye-cloud scan left transmittance domain")
    return PresamplingDyeCloudScanResult(
        reference=reference,
        candidate=candidate,
        wrong_order=wrong_order,
        post_raster_circular=post_circular,
        post_raster_gaussian=np.asarray(post_gaussian, dtype=np.float64),
        unfiltered_target_samples=unfiltered,
        candidate_kernel=candidate_kernel,
        reference_kernel=reference_kernel,
    )


__all__ = [
    "PresamplingDyeCloudScanResult",
    "render_presampling_dye_cloud_scan",
]
