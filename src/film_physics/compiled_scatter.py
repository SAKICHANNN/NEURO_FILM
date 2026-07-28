"""Explicit-float32 separable challenger to the U6.P1 scatter reference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from scipy.ndimage import convolve1d

from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .reference_scatter import (
    ReferenceScatterProfile,
    gaussian_kernel_1d,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CompiledScatterKernel:
    component_id: str
    weights: np.ndarray
    energy_fraction_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("compiled scatter component_id must be non-empty")
        fractions = tuple(float(value) for value in self.energy_fraction_rgb)
        if len(fractions) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in fractions
        ):
            raise ValueError("compiled scatter energy fractions must be finite [0, 1]")
        weights = np.asarray(self.weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size % 2 != 1 or weights.size < 3:
            raise ValueError("compiled scatter weights must be an odd 1D kernel")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("compiled scatter weights must be finite and nonnegative")
        total = np.sum(weights, dtype=np.float32)
        if total <= 0.0:
            raise ValueError("compiled scatter kernel must have positive energy")
        owned = np.asarray(weights / total, dtype=np.float32)
        owned.setflags(write=False)
        object.__setattr__(self, "weights", owned)
        object.__setattr__(self, "energy_fraction_rgb", fractions)

    @property
    def radius(self) -> int:
        return self.weights.size // 2


@dataclass(frozen=True)
class CompiledScatterProfile:
    parent_profile_sha256: str
    pixel_pitch_um: float
    direct_fraction_rgb: tuple[float, float, float]
    kernels: tuple[CompiledScatterKernel, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.parent_profile_sha256, str
        ) or not _SHA256_RE.fullmatch(self.parent_profile_sha256):
            raise ValueError("parent_profile_sha256 must be lowercase hexadecimal")
        if not math.isfinite(self.pixel_pitch_um) or self.pixel_pitch_um <= 0.0:
            raise ValueError("compiled scatter pixel_pitch_um must be positive")
        direct = tuple(float(value) for value in self.direct_fraction_rgb)
        kernels = tuple(self.kernels)
        if len(direct) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in direct
        ):
            raise ValueError("compiled scatter direct fractions must be finite [0, 1]")
        if not kernels or not all(
            isinstance(kernel, CompiledScatterKernel) for kernel in kernels
        ):
            raise ValueError("compiled scatter requires typed kernels")
        if len({kernel.component_id for kernel in kernels}) != len(kernels):
            raise ValueError("compiled scatter component IDs must be unique")
        scatter = np.sum(
            np.asarray(
                [kernel.energy_fraction_rgb for kernel in kernels], dtype=np.float64
            ),
            axis=0,
        )
        if not np.allclose(
            np.asarray(direct) + scatter, 1.0, rtol=0.0, atol=2e-15
        ):
            raise ValueError("compiled scatter energy partition must sum to one")
        object.__setattr__(self, "direct_fraction_rgb", direct)
        object.__setattr__(self, "kernels", kernels)

    @property
    def required_halo(self) -> int:
        return max(kernel.radius for kernel in self.kernels)


def compile_scatter_profile(
    reference: ReferenceScatterProfile,
) -> CompiledScatterProfile:
    kernels = []
    for component in reference.components:
        kernel = gaussian_kernel_1d(component, reference.scale).astype(np.float32)
        kernel /= np.sum(kernel, dtype=np.float32)
        kernels.append(
            CompiledScatterKernel(
                component.component_id,
                kernel,
                component.energy_fraction_rgb,
            )
        )
    return CompiledScatterProfile(
        parent_profile_sha256=reference.profile_sha256,
        pixel_pitch_um=reference.pixel_pitch_um,
        direct_fraction_rgb=tuple(reference.direct_fraction.tolist()),  # type: ignore[arg-type]
        kernels=tuple(kernels),
    )


def apply_compiled_scatter(
    exposure: PhysicalDomainArray,
    profile: CompiledScatterProfile,
) -> PhysicalDomainArray:
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float32:
        raise TypeError("compiled scatter requires float32 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("compiled scatter requires an HxWx3 image")
    if exposure.scale is None or exposure.scale.pixel_pitch_um != profile.pixel_pitch_um:
        raise ValueError("compiled scatter pixel scale does not match the profile")
    values = exposure.values
    direct = np.asarray(profile.direct_fraction_rgb, dtype=np.float32)
    output = np.asarray(values * direct.reshape(1, 1, 3), dtype=np.float32)
    for kernel in profile.kernels:
        fractions = np.asarray(kernel.energy_fraction_rgb, dtype=np.float32)
        for channel in range(3):
            if fractions[channel] == 0.0:
                continue
            horizontal = convolve1d(
                values[..., channel],
                kernel.weights,
                axis=1,
                output=np.float32,
                mode="constant",
                cval=0.0,
            )
            blurred = convolve1d(
                horizontal,
                kernel.weights,
                axis=0,
                output=np.float32,
                mode="constant",
                cval=0.0,
            )
            output[..., channel] += fractions[channel] * blurred
    if np.any(output < 0.0):
        raise RuntimeError("compiled positive scatter produced negative exposure")
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )


def apply_compiled_scatter_row_tiled(
    exposure: PhysicalDomainArray,
    profile: CompiledScatterProfile,
    *,
    tile_rows: int,
) -> PhysicalDomainArray:
    """Execute with exact global-coordinate row halos and stitch the result."""

    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    height = exposure.values.shape[0]
    output = np.empty_like(exposure.values)
    halo = profile.required_halo
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        window = PhysicalDomainArray(
            exposure.values[source_y0:source_y1],
            exposure.domain,
            exposure.unit,
            exposure.channels,
            exposure.scale,
        )
        rendered = apply_compiled_scatter(window, profile).values
        crop_y0 = y0 - source_y0
        output[y0:y1] = rendered[crop_y0 : crop_y0 + (y1 - y0)]
    return PhysicalDomainArray(
        output,
        exposure.domain,
        exposure.unit,
        exposure.channels,
        exposure.scale,
    )
