"""Explicit, ordered spatial-response primitives for physical film imaging."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    if np.any(output > 1.0):
        raise RuntimeError("scanner MTF left scan-linear domain")
    return output
