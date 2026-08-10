"""Positive separable streaming compiler for multiscale scanner glare."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from .scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareDomainError,
)


@dataclass(frozen=True)
class SeparableScannerGlareKernel:
    """Positive normalized 1-D component kernels sharing one support radius."""

    weights: tuple[float, ...]
    kernels: tuple[np.ndarray, ...]
    radius: int

    def __post_init__(self) -> None:
        if not self.weights or len(self.weights) != len(self.kernels):
            raise ScannerGlareDomainError("invalid separable component inventory")
        if self.radius < 1:
            raise ScannerGlareDomainError("separable radius must be positive")
        if not math.isclose(math.fsum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ScannerGlareDomainError("separable weights must sum to one")
        expected = 2 * self.radius + 1
        for weight, kernel in zip(self.weights, self.kernels, strict=True):
            values = np.asarray(kernel)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ScannerGlareDomainError("separable weights must be positive")
            if (
                values.shape != (expected,)
                or values.dtype != np.float64
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or not math.isclose(
                    float(np.sum(values)), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ScannerGlareDomainError("invalid separable component kernel")


def compile_scanner_glare_separable(
    profile: MultiscaleScannerGlareProfile, *, kernel_size: int
) -> SeparableScannerGlareKernel:
    """Compile the same Gaussian mixture as positive normalized 1-D factors."""
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
    kernels = []
    for component in profile.components:
        kernel = np.exp(-0.5 * np.square(coordinates / component.sigma_pixels))
        kernel /= np.sum(kernel)
        kernel.setflags(write=False)
        kernels.append(kernel)
    return SeparableScannerGlareKernel(
        weights=tuple(component.weight for component in profile.components),
        kernels=tuple(kernels),
        radius=radius,
    )


def _symmetric_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size < 1:
        raise ScannerGlareDomainError("spatial dimensions must be positive")
    period = 2 * size
    folded = np.mod(indices, period)
    return np.where(folded < size, folded, period - 1 - folded).astype(np.intp)


def scanner_glare_streaming_workspace_bytes(
    *, width: int, channels: int, row_chunk: int, radius: int
) -> int:
    """Return explicitly tracked peak scratch bytes, excluding caller input/output."""
    if min(width, channels, row_chunk, radius) < 1:
        raise ScannerGlareDomainError("workspace dimensions must be positive")
    strip_values = (row_chunk + 2 * radius) * width * channels
    return 3 * strip_values * np.dtype(np.float64).itemsize


def apply_scanner_glare_separable_streaming(
    transmittance: np.ndarray,
    compiled: SeparableScannerGlareKernel,
    *,
    flare_fraction: float,
    row_chunk: int,
) -> np.ndarray:
    """Apply positive scanner glare in halo-aware row chunks."""
    values = np.asarray(transmittance, dtype=np.float64)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] != 3):
        raise ScannerGlareDomainError("transmittance must be HxW or HxWx3")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ScannerGlareDomainError("transmittance must be finite in [0, 1]")
    if not math.isfinite(flare_fraction) or not 0.0 <= flare_fraction < 1.0:
        raise ScannerGlareDomainError("flare fraction must be in [0, 1)")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk < 1:
        raise ScannerGlareDomainError("row chunk must be a positive integer")
    height = values.shape[0]
    planes = values[..., None] if values.ndim == 2 else values
    output = (1.0 - flare_fraction) * planes.copy()
    radius = compiled.radius
    for weight, kernel in zip(compiled.weights, compiled.kernels, strict=True):
        horizontal_cache: dict[int, np.ndarray] = {}
        for start in range(0, height, row_chunk):
            stop = min(start + row_chunk, height)
            raw_indices = np.arange(start - radius, stop + radius, dtype=np.int64)
            mapped = _symmetric_indices(raw_indices, height)
            required_rows = {int(index) for index in mapped}
            for cached_index in tuple(horizontal_cache):
                if cached_index not in required_rows:
                    del horizontal_cache[cached_index]
            for source_index in required_rows:
                if source_index not in horizontal_cache:
                    horizontal_cache[source_index] = correlate1d(
                        planes[source_index : source_index + 1],
                        kernel,
                        axis=1,
                        mode="reflect",
                    )
            horizontal = np.concatenate(
                [horizontal_cache[int(index)] for index in mapped], axis=0
            )
            vertical = correlate1d(horizontal, kernel, axis=0, mode="nearest")
            output[start:stop] += (
                flare_fraction
                * weight
                * vertical[radius : radius + (stop - start)]
            )
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise RuntimeError("separable scanner glare left bounded transmittance domain")
    bounded = np.clip(output, 0.0, 1.0)
    return bounded[..., 0] if values.ndim == 2 else bounded
