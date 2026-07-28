"""Explicit, ordered spatial-response primitives for physical film imaging."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class SpatialResponseProfile:
    pixel_pitch_um: float
    forward_scatter_sigma_um_rgb: tuple[float, float, float]
    development_adjacency_sigma_um_rgb: tuple[float, float, float]
    development_adjacency_gain_rgb: tuple[float, float, float]
    dye_diffusion_sigma_um_rgb: tuple[float, float, float]
    scanner_mtf_sigma_um_rgb: tuple[float, float, float]
    gaussian_truncate: float = 4.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.pixel_pitch_um) or self.pixel_pitch_um <= 0.0:
            raise ValueError("pixel pitch must be finite and positive")
        groups = (
            self.forward_scatter_sigma_um_rgb,
            self.development_adjacency_sigma_um_rgb,
            self.development_adjacency_gain_rgb,
            self.dye_diffusion_sigma_um_rgb,
            self.scanner_mtf_sigma_um_rgb,
        )
        if any(len(group) != 3 for group in groups):
            raise ValueError("spatial profile requires exactly three channels")
        if any(
            not math.isfinite(value) or value < 0.0
            for group in groups
            for value in group
        ):
            raise ValueError("spatial parameters must be finite and nonnegative")
        if not math.isfinite(self.gaussian_truncate) or self.gaussian_truncate <= 0.0:
            raise ValueError("Gaussian truncate must be finite and positive")


def _validate(values: np.ndarray, *, maximum: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("spatial response requires a HxWx3 array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("spatial response input must be finite and nonnegative")
    if maximum is not None and np.any(array > maximum):
        raise ValueError("spatial response input exceeds its domain maximum")
    return array


def _blur(
    values: np.ndarray,
    sigmas_um: tuple[float, float, float],
    profile: SpatialResponseProfile,
) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for channel, sigma_um in enumerate(sigmas_um):
        sigma_pixels = sigma_um / profile.pixel_pitch_um
        if sigma_pixels == 0.0:
            output[..., channel] = values[..., channel]
        else:
            output[..., channel] = gaussian_filter(
                values[..., channel],
                sigma=sigma_pixels,
                order=0,
                mode="nearest",
                truncate=profile.gaussian_truncate,
            )
    return output


def apply_forward_scatter(
    layer_exposure: np.ndarray, profile: SpatialResponseProfile
) -> np.ndarray:
    exposure = _validate(layer_exposure)
    return _blur(exposure, profile.forward_scatter_sigma_um_rgb, profile)


def apply_development_adjacency(
    developed_density: np.ndarray, profile: SpatialResponseProfile
) -> np.ndarray:
    density = _validate(developed_density)
    blurred = _blur(
        density, profile.development_adjacency_sigma_um_rgb, profile
    )
    gains = np.asarray(profile.development_adjacency_gain_rgb, dtype=np.float64)
    output = density + gains * (density - blurred)
    if np.any(output < 0.0) or not np.all(np.isfinite(output)):
        raise RuntimeError("development adjacency left optical-density domain")
    return output


def apply_bounded_development_adjacency(
    developed_density: np.ndarray,
    profile: SpatialResponseProfile,
    *,
    maximum_absolute_transmittance_delta: float,
    maximum_absolute_density_delta: float,
) -> np.ndarray:
    """Apply a smooth density-domain adjacency correction with analytic bounds."""
    density = _validate(developed_density)
    if (
        not math.isfinite(maximum_absolute_transmittance_delta)
        or maximum_absolute_transmittance_delta <= 0.0
        or maximum_absolute_transmittance_delta >= 1.0
    ):
        raise ValueError("transmittance bound must be finite and inside (0, 1)")
    if (
        not math.isfinite(maximum_absolute_density_delta)
        or maximum_absolute_density_delta <= 0.0
    ):
        raise ValueError("density bound must be finite and positive")
    blurred = _blur(
        density, profile.development_adjacency_sigma_um_rgb, profile
    )
    gains = np.asarray(profile.development_adjacency_gain_rgb, dtype=np.float64)
    raw = gains * (density - blurred)
    transmittance = np.power(10.0, -density)
    lower_transmittance = np.maximum(
        transmittance - maximum_absolute_transmittance_delta,
        np.finfo(np.float64).tiny,
    )
    upper_transmittance = np.minimum(
        transmittance + maximum_absolute_transmittance_delta, 1.0
    )
    positive_limit = np.minimum(
        -np.log10(lower_transmittance) - density,
        maximum_absolute_density_delta,
    )
    negative_limit = np.minimum(
        density + np.log10(upper_transmittance),
        maximum_absolute_density_delta,
    )
    limit = np.where(raw >= 0.0, positive_limit, negative_limit)
    correction = np.zeros_like(raw)
    active = limit > 0.0
    correction[active] = (
        np.sign(raw[active])
        * limit[active]
        * np.tanh(np.abs(raw[active]) / limit[active])
    )
    output = density + correction
    if np.any(output < 0.0) or not np.all(np.isfinite(output)):
        raise RuntimeError("bounded development adjacency left density domain")
    output_transmittance = np.power(10.0, -output)
    if (
        np.max(np.abs(output_transmittance - transmittance))
        > maximum_absolute_transmittance_delta + 1e-12
    ):
        raise RuntimeError("bounded development adjacency violated transmittance bound")
    return output


def apply_interpretation_bounded_development_adjacency(
    developed_density: np.ndarray,
    profile: SpatialResponseProfile,
    *,
    maximum_absolute_transmittance_delta: float,
    maximum_absolute_density_delta: float,
    black_reference_density: np.ndarray,
    white_reference_density: np.ndarray,
) -> np.ndarray:
    """Apply P5C adjacency while preserving an interpretation density domain."""
    density = _validate(developed_density)
    black = np.asarray(black_reference_density, dtype=np.float64)
    white = np.asarray(white_reference_density, dtype=np.float64)
    if (
        black.shape != (3,)
        or white.shape != (3,)
        or not np.all(np.isfinite(black))
        or not np.all(np.isfinite(white))
        or np.any(black < 0.0)
        or np.any(white <= black)
    ):
        raise ValueError("interpretation references must be finite ordered RGB")
    if np.any(density < black - 1e-12) or np.any(density > white + 1e-12):
        raise ValueError("input density falls outside interpretation references")
    if (
        not math.isfinite(maximum_absolute_transmittance_delta)
        or maximum_absolute_transmittance_delta <= 0.0
        or maximum_absolute_transmittance_delta >= 1.0
    ):
        raise ValueError("transmittance bound must be finite and inside (0, 1)")
    if (
        not math.isfinite(maximum_absolute_density_delta)
        or maximum_absolute_density_delta <= 0.0
    ):
        raise ValueError("density bound must be finite and positive")

    blurred = _blur(
        density, profile.development_adjacency_sigma_um_rgb, profile
    )
    gains = np.asarray(profile.development_adjacency_gain_rgb, dtype=np.float64)
    raw = gains * (density - blurred)
    transmittance = np.power(10.0, -density)
    lower_transmittance = np.maximum(
        transmittance - maximum_absolute_transmittance_delta,
        np.finfo(np.float64).tiny,
    )
    upper_transmittance = np.minimum(
        transmittance + maximum_absolute_transmittance_delta, 1.0
    )
    positive_limit = np.minimum.reduce(
        (
            -np.log10(lower_transmittance) - density,
            np.full_like(density, maximum_absolute_density_delta),
            white.reshape(1, 1, 3) - density,
        )
    )
    negative_limit = np.minimum.reduce(
        (
            density + np.log10(upper_transmittance),
            np.full_like(density, maximum_absolute_density_delta),
            density - black.reshape(1, 1, 3),
        )
    )
    limit = np.where(raw >= 0.0, positive_limit, negative_limit)
    correction = np.zeros_like(raw)
    active = limit > 0.0
    correction[active] = (
        np.sign(raw[active])
        * limit[active]
        * np.tanh(np.abs(raw[active]) / limit[active])
    )
    output = density + correction
    if (
        not np.all(np.isfinite(output))
        or np.any(output < black - 1e-12)
        or np.any(output > white + 1e-12)
    ):
        raise RuntimeError("adjacency escaped interpretation density domain")
    output_transmittance = np.power(10.0, -output)
    if (
        np.max(np.abs(output_transmittance - transmittance))
        > maximum_absolute_transmittance_delta + 1e-12
    ):
        raise RuntimeError("adjacency violated transmittance bound")
    return output


def apply_dye_diffusion(
    developed_density: np.ndarray, profile: SpatialResponseProfile
) -> np.ndarray:
    density = _validate(developed_density)
    return _blur(density, profile.dye_diffusion_sigma_um_rgb, profile)


def density_to_scan_transmittance(
    developed_density: np.ndarray,
) -> np.ndarray:
    density = _validate(developed_density)
    output = np.power(10.0, -density)
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0) or np.any(output > 1.0):
        raise RuntimeError("density interpretation left transmittance domain")
    return output


def apply_scanner_mtf(
    scan_linear: np.ndarray, profile: SpatialResponseProfile
) -> np.ndarray:
    values = _validate(scan_linear, maximum=1.0)
    output = _blur(values, profile.scanner_mtf_sigma_um_rgb, profile)
    tolerance = 1e-12
    if np.any(output > 1.0 + tolerance):
        raise RuntimeError("scanner MTF left scan-linear domain")
    return np.where(output > 1.0, 1.0, output)


def required_spatial_response_halo(profile: SpatialResponseProfile) -> int:
    """Return the exact summed finite support for the ordered Gaussian stages."""

    def radius(sigmas_um: tuple[float, float, float]) -> int:
        sigma_pixels = max(sigmas_um) / profile.pixel_pitch_um
        return int(profile.gaussian_truncate * sigma_pixels + 0.5)

    return (
        radius(profile.forward_scatter_sigma_um_rgb)
        + radius(profile.development_adjacency_sigma_um_rgb)
        + radius(profile.dye_diffusion_sigma_um_rgb)
        + radius(profile.scanner_mtf_sigma_um_rgb)
    )


def apply_spatial_response_pipeline(
    layer_exposure: np.ndarray,
    profile: SpatialResponseProfile,
    *,
    sensitometry_apply: Callable[[np.ndarray], np.ndarray],
    adjacency_apply: Callable[
        [np.ndarray, SpatialResponseProfile], np.ndarray
    ] = apply_development_adjacency,
) -> np.ndarray:
    """Execute the canonical exposure-to-scan spatial chain."""
    exposure = _validate(layer_exposure)
    scattered = apply_forward_scatter(exposure, profile)
    density = _validate(sensitometry_apply(scattered))
    adjacent = adjacency_apply(density, profile)
    diffused = apply_dye_diffusion(adjacent, profile)
    scan_linear = density_to_scan_transmittance(diffused)
    return apply_scanner_mtf(scan_linear, profile)


def apply_spatial_response_pipeline_row_tiled(
    layer_exposure: np.ndarray,
    profile: SpatialResponseProfile,
    *,
    sensitometry_apply: Callable[[np.ndarray], np.ndarray],
    adjacency_apply: Callable[
        [np.ndarray, SpatialResponseProfile], np.ndarray
    ] = apply_development_adjacency,
    tile_rows: int,
) -> np.ndarray:
    """Execute exact row partitions with the summed finite spatial halo."""
    exposure = _validate(layer_exposure)
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    height = exposure.shape[0]
    halo = required_spatial_response_halo(profile)
    output = np.empty_like(exposure, dtype=np.float64)
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        rendered = apply_spatial_response_pipeline(
            exposure[source_y0:source_y1],
            profile,
            sensitometry_apply=sensitometry_apply,
            adjacency_apply=adjacency_apply,
        )
        output[y0:y1] = rendered[
            y0 - source_y0 : y1 - source_y0
        ]
    return output
