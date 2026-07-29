"""Explicit-float32 separable compiler for bounded backing return."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from scipy.ndimage import convolve1d

from .backing_return import BackingReturnProfile, backing_return_kernel_1d
from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CompiledBackingReturnKernel:
    component_id: str
    weights: np.ndarray
    return_weights: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("compiled backing-return component_id must be non-empty")
        weights = np.asarray(self.weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size < 3 or weights.size % 2 != 1:
            raise ValueError("compiled backing-return weights must be an odd 1D kernel")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "compiled backing-return weights must be finite and nonnegative"
            )
        total = np.sum(weights, dtype=np.float32)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("compiled backing-return kernel must have positive energy")
        owned_weights = np.asarray(weights / total, dtype=np.float32)
        owned_weights.setflags(write=False)

        return_weights = np.asarray(self.return_weights, dtype=np.float32)
        if return_weights.shape != (3, 3):
            raise ValueError("compiled backing-return coupling must be 3x3")
        if not np.all(np.isfinite(return_weights)) or np.any(return_weights < 0.0):
            raise ValueError(
                "compiled backing-return coupling must be finite and nonnegative"
            )
        if np.any(np.sum(return_weights, axis=0, dtype=np.float64) > 1.0):
            raise ValueError(
                "compiled backing-return component exceeds incident source energy"
            )
        owned_return = np.array(return_weights, dtype=np.float32, copy=True)
        owned_return.setflags(write=False)
        object.__setattr__(self, "weights", owned_weights)
        object.__setattr__(self, "return_weights", owned_return)

    @property
    def radius(self) -> int:
        return self.weights.size // 2


@dataclass(frozen=True)
class CompiledBackingReturnProfile:
    parent_profile_sha256: str
    pixel_pitch_um: float
    kernels: tuple[CompiledBackingReturnKernel, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.parent_profile_sha256, str
        ) or not _SHA256_RE.fullmatch(self.parent_profile_sha256):
            raise ValueError("parent_profile_sha256 must be lowercase hexadecimal")
        if not math.isfinite(self.pixel_pitch_um) or self.pixel_pitch_um <= 0.0:
            raise ValueError("compiled backing-return pixel_pitch_um must be positive")
        kernels = tuple(self.kernels)
        if not kernels or not all(
            isinstance(kernel, CompiledBackingReturnKernel) for kernel in kernels
        ):
            raise ValueError("compiled backing return requires typed kernels")
        if len({kernel.component_id for kernel in kernels}) != len(kernels):
            raise ValueError("compiled backing-return component IDs must be unique")
        aggregate = np.sum(
            np.asarray([kernel.return_weights for kernel in kernels], dtype=np.float64),
            axis=0,
        )
        if np.any(np.sum(aggregate, axis=0, dtype=np.float64) > 1.0):
            raise ValueError("aggregate compiled backing return exceeds incident energy")
        object.__setattr__(self, "kernels", kernels)

    @property
    def required_halo(self) -> int:
        return max(kernel.radius for kernel in self.kernels)


def compile_backing_return_profile(
    reference: BackingReturnProfile,
) -> CompiledBackingReturnProfile:
    kernels = []
    for component in reference.components:
        weights = backing_return_kernel_1d(component, reference.scale).astype(
            np.float32
        )
        weights /= np.sum(weights, dtype=np.float32)
        kernels.append(
            CompiledBackingReturnKernel(
                component_id=component.component_id,
                weights=weights,
                return_weights=component.return_weights.astype(np.float32),
            )
        )
    return CompiledBackingReturnProfile(
        parent_profile_sha256=reference.profile_sha256,
        pixel_pitch_um=reference.pixel_pitch_um,
        kernels=tuple(kernels),
    )


def apply_compiled_backing_return(
    exposure: PhysicalDomainArray,
    profile: CompiledBackingReturnProfile,
) -> PhysicalDomainArray:
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float32:
        raise TypeError("compiled backing return requires float32 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("compiled backing return requires an HxWx3 image")
    if exposure.scale is None or exposure.scale.pixel_pitch_um != profile.pixel_pitch_um:
        raise ValueError("compiled backing-return pixel scale does not match profile")

    values = exposure.values
    output = np.array(values, dtype=np.float32, copy=True, order="C")
    for kernel in profile.kernels:
        for source_channel in range(3):
            source_weights = kernel.return_weights[:, source_channel]
            if not np.any(source_weights):
                continue
            horizontal = convolve1d(
                values[..., source_channel],
                kernel.weights,
                axis=1,
                output=np.float32,
                mode="constant",
                cval=0.0,
            )
            returned = convolve1d(
                horizontal,
                kernel.weights,
                axis=0,
                output=np.float32,
                mode="constant",
                cval=0.0,
            )
            for target_layer in range(3):
                weight = source_weights[target_layer]
                if weight != 0.0:
                    output[..., target_layer] += weight * returned
    if np.any(output < values):
        raise RuntimeError("compiled backing return subtracted direct exposure")
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )


def apply_compiled_backing_return_row_tiled(
    exposure: PhysicalDomainArray,
    profile: CompiledBackingReturnProfile,
    *,
    tile_rows: int,
) -> PhysicalDomainArray:
    """Execute exact global-coordinate row windows with finite-support halos."""

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
        rendered = apply_compiled_backing_return(window, profile).values
        crop_y0 = y0 - source_y0
        output[y0:y1] = rendered[crop_y0 : crop_y0 + (y1 - y0)]
    return PhysicalDomainArray(
        output,
        exposure.domain,
        exposure.unit,
        exposure.channels,
        exposure.scale,
    )
