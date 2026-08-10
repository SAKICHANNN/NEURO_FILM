"""Positive energy-normalized multiscale scanner-glare primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


class ScannerGlareDomainError(ValueError):
    """Raised when a scanner-glare profile or signal violates its domain."""


@dataclass(frozen=True)
class ScannerGlareComponent:
    weight: float
    sigma_pixels: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ScannerGlareDomainError("component weight must be positive")
        if not math.isfinite(self.sigma_pixels) or self.sigma_pixels <= 0.0:
            raise ScannerGlareDomainError("component sigma must be positive")


@dataclass(frozen=True)
class MultiscaleScannerGlareProfile:
    components: tuple[ScannerGlareComponent, ...]
    flare_fraction: float
    truncate_sigma: float

    def __post_init__(self) -> None:
        if not self.components:
            raise ScannerGlareDomainError("at least one component is required")
        weight_sum = math.fsum(component.weight for component in self.components)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ScannerGlareDomainError("component weights must sum to one")
        if (
            not math.isfinite(self.flare_fraction)
            or not 0.0 <= self.flare_fraction < 1.0
        ):
            raise ScannerGlareDomainError("flare fraction must be in [0, 1)")
        if not math.isfinite(self.truncate_sigma) or self.truncate_sigma < 3.0:
            raise ScannerGlareDomainError("truncate sigma must be finite and >= 3")

    @property
    def second_moment_sigma_pixels(self) -> float:
        return math.sqrt(
            math.fsum(
                component.weight * component.sigma_pixels**2
                for component in self.components
            )
        )


def compile_scanner_glare_kernel(
    profile: MultiscaleScannerGlareProfile, *, kernel_size: int
) -> np.ndarray:
    """Compile a centered positive 2-D Gaussian-mixture glare spread."""
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ScannerGlareDomainError("kernel size must be an odd integer >= 3")
    radius = kernel_size // 2
    required = math.ceil(
        max(component.sigma_pixels for component in profile.components)
        * profile.truncate_sigma
    )
    if radius < required:
        raise ScannerGlareDomainError(
            "kernel support is smaller than profile truncation"
        )
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    squared_radius = xx * xx + yy * yy
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    for component in profile.components:
        component_kernel = np.exp(-0.5 * squared_radius / (component.sigma_pixels**2))
        component_kernel /= np.sum(component_kernel)
        kernel += component.weight * component_kernel
    kernel /= np.sum(kernel)
    return kernel


def scanner_glare_second_moment(kernel: np.ndarray) -> float:
    """Return per-axis second moment sigma for a centered normalized kernel."""
    values = np.asarray(kernel, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] % 2 == 0
    ):
        raise ScannerGlareDomainError("kernel must be odd square")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ScannerGlareDomainError("kernel must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ScannerGlareDomainError("kernel must have positive energy")
    radius = values.shape[0] // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    _, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return math.sqrt(float(np.sum(values * np.square(xx)) / total))


def apply_scanner_glare(
    transmittance: np.ndarray,
    kernel: np.ndarray,
    *,
    flare_fraction: float,
) -> np.ndarray:
    """Apply positive context-dependent glare with symmetric finite support."""
    values = np.asarray(transmittance, dtype=np.float64)
    spread = np.asarray(kernel, dtype=np.float64)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] != 3):
        raise ScannerGlareDomainError("transmittance must be HxW or HxWx3")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ScannerGlareDomainError("transmittance must be finite in [0, 1]")
    if not math.isfinite(flare_fraction) or not 0.0 <= flare_fraction < 1.0:
        raise ScannerGlareDomainError("flare fraction must be in [0, 1)")
    if (
        spread.ndim != 2
        or spread.shape[0] != spread.shape[1]
        or spread.shape[0] % 2 == 0
    ):
        raise ScannerGlareDomainError("kernel must be odd square")
    if not np.all(np.isfinite(spread)) or np.any(spread < 0.0):
        raise ScannerGlareDomainError("kernel must be finite and nonnegative")
    if not math.isclose(float(np.sum(spread)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ScannerGlareDomainError("kernel must sum to one")
    radius = spread.shape[0] // 2

    def convolve(plane: np.ndarray) -> np.ndarray:
        padded = np.pad(plane, radius, mode="symmetric")
        return fftconvolve(padded, spread, mode="valid")

    if values.ndim == 2:
        blurred = convolve(values)
    else:
        blurred = np.stack(
            [convolve(values[..., channel]) for channel in range(3)], axis=-1
        )
    output = (1.0 - flare_fraction) * values + flare_fraction * blurred
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise RuntimeError("scanner glare left bounded transmittance domain")
    return np.clip(output, 0.0, 1.0)
