"""Analytical, energy-normalized support-return geometry reference.

This module models only the spatial topology implied by a rear support
reflection. Its parameters are explicit generic hypotheses, not measurements
of a named film stock.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class BaseReturnGeometryProfile:
    support_thickness_um: float
    refractive_index: float
    attenuation_length_um: float
    maximum_radius_um: float
    pixel_pitch_um: float

    def __post_init__(self) -> None:
        values = (
            self.support_thickness_um,
            self.refractive_index,
            self.attenuation_length_um,
            self.maximum_radius_um,
            self.pixel_pitch_um,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("base-return geometry values must be finite")
        if self.support_thickness_um <= 0.0:
            raise ValueError("support thickness must be positive")
        if self.refractive_index <= 1.0:
            raise ValueError("support refractive index must exceed one")
        if self.attenuation_length_um <= 0.0:
            raise ValueError("attenuation length must be positive")
        if self.pixel_pitch_um <= 0.0:
            raise ValueError("pixel pitch must be positive")
        if self.maximum_radius_um <= self.critical_radius_um:
            raise ValueError("maximum radius must exceed the critical radius")

    @property
    def critical_angle_rad(self) -> float:
        return math.asin(1.0 / self.refractive_index)

    @property
    def critical_radius_um(self) -> float:
        return 2.0 * self.support_thickness_um * math.tan(self.critical_angle_rad)


@dataclass(frozen=True)
class BaseReturnSpread:
    output: np.ndarray
    residual: np.ndarray


def _radius_grid(profile: BaseReturnGeometryProfile) -> np.ndarray:
    radius_px = int(math.ceil(profile.maximum_radius_um / profile.pixel_pitch_um))
    axis = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.hypot(xx, yy) * profile.pixel_pitch_um


def base_return_geometry_kernel(profile: BaseReturnGeometryProfile) -> np.ndarray:
    """Rasterize the isotropic TIR ray-angle pushforward as a 2D kernel."""

    radius = _radius_grid(profile)
    two_t = 2.0 * profile.support_thickness_um
    theta = np.arctan(radius / two_t)
    valid = (
        (radius >= profile.critical_radius_um)
        & (radius <= profile.maximum_radius_um)
    )
    density = np.zeros_like(radius)
    if np.any(valid):
        selected_radius = radius[valid]
        selected_theta = theta[valid]
        angular_flux = 2.0 * np.sin(selected_theta) * np.cos(selected_theta)
        path = two_t / np.cos(selected_theta)
        attenuation = np.exp(-path / profile.attenuation_length_um)
        dtheta_dr = two_t / (np.square(selected_radius) + two_t * two_t)
        density[valid] = (
            angular_flux
            * attenuation
            * dtheta_dr
            / (2.0 * math.pi * selected_radius)
        )
    total = float(np.sum(density, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("base-return geometry has no rasterized support")
    density /= total
    density.setflags(write=False)
    return density


def equal_second_moment_gaussian(
    candidate: np.ndarray, pixel_pitch_um: float
) -> np.ndarray:
    """Return a finite-support Gaussian with candidate-matched radial moment."""

    kernel = np.asarray(candidate, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("candidate kernel must be odd square")
    if pixel_pitch_um <= 0.0 or not math.isfinite(pixel_pitch_um):
        raise ValueError("pixel pitch must be finite and positive")
    radius_px = kernel.shape[0] // 2
    axis = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius2 = (np.square(xx) + np.square(yy)) * pixel_pitch_um**2
    target = float(np.sum(kernel * radius2, dtype=np.float64))

    low = pixel_pitch_um / 100.0
    high = max(pixel_pitch_um, radius_px * pixel_pitch_um * 10.0)
    for _ in range(96):
        sigma = 0.5 * (low + high)
        trial = np.exp(-0.5 * radius2 / (sigma * sigma))
        trial /= np.sum(trial, dtype=np.float64)
        moment = float(np.sum(trial * radius2, dtype=np.float64))
        if moment < target:
            low = sigma
        else:
            high = sigma
    sigma = 0.5 * (low + high)
    gaussian = np.exp(-0.5 * radius2 / (sigma * sigma))
    gaussian /= np.sum(gaussian, dtype=np.float64)
    gaussian.setflags(write=False)
    return gaussian


def radial_quantile(
    kernel: np.ndarray, pixel_pitch_um: float, quantile: float
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    values = np.asarray(kernel, dtype=np.float64)
    radius_px = values.shape[0] // 2
    axis = np.arange(-radius_px, radius_px + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy).reshape(-1) * pixel_pitch_um
    weights = values.reshape(-1)
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return float(radius[order[index]])


def _symmetric_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="symmetric")
    convolved = fftconvolve(padded, kernel, mode="same")
    return convolved[radius : radius + values.shape[0], radius : radius + values.shape[1]]


def apply_positive_base_return_geometry(
    exposure: np.ndarray,
    kernel: np.ndarray,
    return_fraction_rgb: tuple[float, float, float],
) -> BaseReturnSpread:
    """Add only exterior spread from the normalized support-return topology."""

    values = np.asarray(exposure)
    if values.dtype != np.float64 or values.ndim != 3 or values.shape[-1] != 3:
        raise TypeError("base-return spread requires HxWx3 float64 exposure")
    fractions = np.asarray(return_fraction_rgb, dtype=np.float64)
    if fractions.shape != (3,) or np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("return fractions must be three finite values in [0, 1]")
    candidate = np.asarray(kernel, dtype=np.float64)
    if candidate.ndim != 2 or np.any(candidate < 0.0) or not np.isclose(np.sum(candidate), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("base-return kernel must be normalized and nonnegative")
    residual = np.zeros_like(values)
    tolerance = 64.0 * np.finfo(np.float64).eps
    for channel in range(3):
        blurred = _symmetric_convolve(values[..., channel], candidate)
        positive = np.where(blurred - values[..., channel] > tolerance, blurred - values[..., channel], 0.0)
        residual[..., channel] = fractions[channel] * positive
    output = values + residual
    if np.any(~np.isfinite(output)) or np.any(output < values):
        raise RuntimeError("base-return spread left its additive exposure domain")
    return BaseReturnSpread(output=output, residual=residual)


__all__ = [
    "BaseReturnGeometryProfile",
    "BaseReturnSpread",
    "apply_positive_base_return_geometry",
    "base_return_geometry_kernel",
    "equal_second_moment_gaussian",
    "radial_quantile",
]
