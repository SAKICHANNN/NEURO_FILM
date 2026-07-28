"""Shape-stable near-direct/far-pyramid scatter challenger."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .compiled_scatter import (
    CompiledScatterKernel,
    CompiledScatterProfile,
    apply_compiled_scatter,
)
from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .reference_scatter import ReferenceScatterProfile, gaussian_kernel_1d


@dataclass(frozen=True)
class PyramidScatterComponent:
    component_id: str
    sigma_pixels: float
    energy_fraction_rgb: tuple[float, float, float]
    direct_kernel: CompiledScatterKernel | None
    pyramid_factor: int
    cutoff_sigma: float

    @property
    def uses_pyramid(self) -> bool:
        return self.direct_kernel is None


@dataclass(frozen=True)
class PyramidScatterProfile:
    parent_profile_sha256: str
    pixel_pitch_um: float
    direct_fraction_rgb: tuple[float, float, float]
    components: tuple[PyramidScatterComponent, ...]
    target_coarse_sigma_pixels: float


def compile_pyramid_scatter_profile(
    reference: ReferenceScatterProfile,
    *,
    target_coarse_sigma_pixels: float,
    minimum_pyramid_factor: int,
) -> PyramidScatterProfile:
    target = float(target_coarse_sigma_pixels)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target coarse sigma must be finite and positive")
    if (
        isinstance(minimum_pyramid_factor, bool)
        or not isinstance(minimum_pyramid_factor, int)
        or minimum_pyramid_factor < 2
    ):
        raise ValueError("minimum pyramid factor must be an integer >= 2")
    components = []
    for component in reference.components:
        sigma_pixels = component.sigma_um / reference.pixel_pitch_um
        factor = int(math.floor(sigma_pixels / target))
        if factor < minimum_pyramid_factor:
            weights = gaussian_kernel_1d(component, reference.scale).astype(np.float32)
            weights /= np.sum(weights, dtype=np.float32)
            direct = CompiledScatterKernel(
                component.component_id,
                weights,
                component.energy_fraction_rgb,
            )
            factor = 1
        else:
            direct = None
        components.append(
            PyramidScatterComponent(
                component.component_id,
                sigma_pixels,
                component.energy_fraction_rgb,
                direct,
                factor,
                component.cutoff_sigma,
            )
        )
    return PyramidScatterProfile(
        reference.profile_sha256,
        reference.pixel_pitch_um,
        tuple(reference.direct_fraction.tolist()),  # type: ignore[arg-type]
        tuple(components),
        target,
    )


def _area_downsample_axis(
    array: np.ndarray, destination: int, axis: int
) -> np.ndarray:
    source = array.shape[axis]
    if source == destination:
        return array.astype(np.float32, copy=True)
    moved = np.moveaxis(array, axis, 0)
    output = np.empty((destination,) + moved.shape[1:], dtype=np.float32)
    scale = source / destination
    for index in range(destination):
        start = index * scale
        end = (index + 1) * scale
        first = int(math.floor(start))
        stop = int(math.ceil(end))
        cells = np.arange(first, stop, dtype=np.float64)
        weights = np.minimum(end, cells + 1.0) - np.maximum(start, cells)
        weights = (weights / (end - start)).astype(np.float32)
        accumulator = np.zeros(moved.shape[1:], dtype=np.float32)
        for source_index, weight in zip(range(first, stop), weights, strict=True):
            accumulator += moved[source_index] * weight
        output[index] = accumulator
    return np.moveaxis(output, 0, axis)


def _area_downsample(channel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    horizontal = _area_downsample_axis(channel, shape[1], axis=1)
    return _area_downsample_axis(horizontal, shape[0], axis=0)


def _bilinear_reconstruct(channel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    coarse_height, coarse_width = channel.shape
    ys = (np.arange(height, dtype=np.float64) + 0.5) * coarse_height / height - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * coarse_width / width - 0.5
    y_floor = np.floor(ys).astype(np.int64)
    x_floor = np.floor(xs).astype(np.int64)
    wy = (ys - y_floor).astype(np.float32)
    wx = (xs - x_floor).astype(np.float32)
    y0 = np.clip(y_floor, 0, coarse_height - 1)
    y1 = np.clip(y_floor + 1, 0, coarse_height - 1)
    x0 = np.clip(x_floor, 0, coarse_width - 1)
    x1 = np.clip(x_floor + 1, 0, coarse_width - 1)
    output = np.empty((height, width), dtype=np.float32)
    for row in range(height):
        top = channel[y0[row], x0] * (1.0 - wx) + channel[y0[row], x1] * wx
        bottom = channel[y1[row], x0] * (1.0 - wx) + channel[y1[row], x1] * wx
        output[row] = top * np.float32(1.0 - wy[row]) + bottom * wy[row]
    return output


def _pyramid_blur(
    channel: np.ndarray,
    *,
    sigma_pixels: float,
    factor: int,
    cutoff_sigma: float,
) -> np.ndarray:
    height, width = channel.shape
    coarse_shape = (max(2, height // factor), max(2, width // factor))
    scale_y = height / coarse_shape[0]
    scale_x = width / coarse_shape[1]
    coarse = _area_downsample(channel, coarse_shape)
    blurred = gaussian_filter(
        coarse,
        sigma=(sigma_pixels / scale_y, sigma_pixels / scale_x),
        order=0,
        output=np.float32,
        mode="constant",
        cval=0.0,
        truncate=cutoff_sigma,
    )
    return _bilinear_reconstruct(blurred, (height, width))


def apply_pyramid_scatter(
    exposure: PhysicalDomainArray,
    profile: PyramidScatterProfile,
) -> PhysicalDomainArray:
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float32:
        raise TypeError("pyramid scatter requires float32 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("pyramid scatter requires an HxWx3 image")
    if exposure.scale is None or exposure.scale.pixel_pitch_um != profile.pixel_pitch_um:
        raise ValueError("pyramid scatter pixel scale does not match the profile")
    values = exposure.values
    direct_fraction = np.asarray(profile.direct_fraction_rgb, dtype=np.float32)
    output = np.asarray(values * direct_fraction.reshape(1, 1, 3), dtype=np.float32)
    for component in profile.components:
        fractions = np.asarray(component.energy_fraction_rgb, dtype=np.float32)
        if component.direct_kernel is not None:
            isolated = CompiledScatterProfile(
                profile.parent_profile_sha256,
                profile.pixel_pitch_um,
                tuple(
                    1.0 - value for value in component.energy_fraction_rgb
                ),  # type: ignore[arg-type]
                (component.direct_kernel,),
            )
            component_output = apply_compiled_scatter(exposure, isolated).values
            blurred = (
                component_output
                - values * (1.0 - fractions).reshape(1, 1, 3)
            ) / np.maximum(fractions.reshape(1, 1, 3), np.float32(1e-30))
            for channel in range(3):
                output[..., channel] += fractions[channel] * blurred[..., channel]
        else:
            for channel in range(3):
                if fractions[channel] == 0.0:
                    continue
                blurred = _pyramid_blur(
                    values[..., channel],
                    sigma_pixels=component.sigma_pixels,
                    factor=component.pyramid_factor,
                    cutoff_sigma=component.cutoff_sigma,
                )
                output[..., channel] += fractions[channel] * blurred
    if np.any(output < 0.0):
        raise RuntimeError("pyramid positive scatter produced negative exposure")
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )
