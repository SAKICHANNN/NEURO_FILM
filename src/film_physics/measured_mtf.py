"""Compact positive-PSF representation for measured film MTF approximations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import ndtr

BUNDLE_SCHEMA = "neuro_film.measured_positive_psf_bundle.v1"


@dataclass(frozen=True)
class PositivePsfComponent:
    weight: float
    sigma_um: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.weight)
            or not math.isfinite(self.sigma_um)
            or self.weight <= 0.0
            or self.sigma_um < 0.0
        ):
            raise ValueError("positive PSF component must be finite and nonnegative")


@dataclass(frozen=True)
class ChannelPsf:
    family: str
    components: tuple[PositivePsfComponent, ...]

    def __post_init__(self) -> None:
        if self.family not in {"single_gaussian", "delta_plus_gaussian", "two_gaussian"}:
            raise ValueError("unsupported positive PSF family")
        if not self.components or abs(sum(row.weight for row in self.components) - 1.0) > 1e-12:
            raise ValueError("positive PSF weights must sum to one")
        sigmas = tuple(row.sigma_um for row in self.components)
        if tuple(sorted(sigmas)) != sigmas:
            raise ValueError("positive PSF sigmas must be ordered")
        expected_count = 1 if self.family == "single_gaussian" else 2
        if len(self.components) != expected_count:
            raise ValueError("positive PSF component count does not match family")
        if self.family == "delta_plus_gaussian" and self.components[0].sigma_um != 0.0:
            raise ValueError("delta-plus-Gaussian requires an exact zero-sigma component")
        if self.family == "two_gaussian" and self.components[0].sigma_um <= 0.0:
            raise ValueError("two-Gaussian components must both have positive sigma")

    def response(self, frequencies_cycles_per_mm: Sequence[float]) -> np.ndarray:
        frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
            raise ValueError("MTF frequencies must be a finite nonnegative vector")
        result = np.zeros_like(frequencies)
        for component in self.components:
            sigma_mm = component.sigma_um * 1e-3
            result += component.weight * np.exp(
                -2.0 * math.pi**2 * sigma_mm**2 * frequencies**2
            )
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "components": [
                {"weight": row.weight, "sigma_um": row.sigma_um}
                for row in self.components
            ],
        }


@dataclass(frozen=True)
class CompiledPsfComponent:
    weight: float
    kernel_1d: np.ndarray

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel_1d, dtype=np.float64)
        if (
            kernel.ndim != 1
            or kernel.size % 2 != 1
            or not np.all(np.isfinite(kernel))
            or np.any(kernel < 0.0)
            or abs(float(np.sum(kernel)) - 1.0) > 1e-14
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise ValueError("compiled PSF component must be positive and normalized")
        object.__setattr__(self, "kernel_1d", kernel)

    @property
    def radius(self) -> int:
        return self.kernel_1d.size // 2


def compile_channel_psf(
    model: ChannelPsf, *, pixel_pitch_um: float, truncate_sigma: float
) -> tuple[CompiledPsfComponent, ...]:
    if (
        not math.isfinite(pixel_pitch_um)
        or pixel_pitch_um <= 0.0
        or not math.isfinite(truncate_sigma)
        or truncate_sigma <= 0.0
    ):
        raise ValueError("PSF compilation scale must be finite and positive")
    compiled = []
    for component in model.components:
        if component.sigma_um == 0.0:
            kernel = np.ones(1, dtype=np.float64)
        else:
            radius = math.ceil(
                truncate_sigma * component.sigma_um / pixel_pitch_um + 0.5
            )
            offsets = np.arange(-radius, radius + 1, dtype=np.float64)
            upper = (offsets + 0.5) * pixel_pitch_um / component.sigma_um
            lower = (offsets - 0.5) * pixel_pitch_um / component.sigma_um
            kernel = ndtr(upper) - ndtr(lower)
            kernel /= np.sum(kernel)
        compiled.append(CompiledPsfComponent(component.weight, kernel))
    return tuple(compiled)


def compiled_channel_response(
    components: Sequence[CompiledPsfComponent],
    frequencies_cycles_per_mm: Sequence[float],
    *,
    pixel_pitch_um: float,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError("compiled MTF frequencies must be finite")
    result = np.zeros_like(frequencies)
    for component in components:
        offsets = np.arange(-component.radius, component.radius + 1, dtype=np.float64)
        phase = (
            2.0
            * math.pi
            * frequencies[:, None]
            * (pixel_pitch_um * 1e-3)
            * offsets[None, :]
        )
        result += component.weight * np.sum(
            component.kernel_1d[None, :] * np.cos(phase), axis=1
        )
    return result


def required_compiled_psf_halo(
    compiled_rgb: Sequence[Sequence[CompiledPsfComponent]],
) -> int:
    return max(
        (component.radius for channel in compiled_rgb for component in channel),
        default=0,
    )


def apply_compiled_positive_psf(
    values: np.ndarray,
    compiled_rgb: Sequence[Sequence[CompiledPsfComponent]],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise ValueError("compiled positive PSF requires finite nonnegative HxWx3")
    if len(compiled_rgb) != 3:
        raise ValueError("compiled positive PSF requires three channels")
    output = np.zeros_like(array)
    for channel, components in enumerate(compiled_rgb):
        if not components:
            raise ValueError("compiled positive PSF channel is empty")
        for component in components:
            plane = array[..., channel]
            if component.radius:
                plane = convolve1d(
                    plane, component.kernel_1d, axis=0, mode="nearest"
                )
                plane = convolve1d(
                    plane, component.kernel_1d, axis=1, mode="nearest"
                )
            output[..., channel] += component.weight * plane
    if not np.all(np.isfinite(output)) or np.any(output < -1e-15):
        raise RuntimeError("compiled positive PSF left its nonnegative domain")
    return np.maximum(output, 0.0)


def apply_compiled_positive_psf_row_tiled(
    values: np.ndarray,
    compiled_rgb: Sequence[Sequence[CompiledPsfComponent]],
    *,
    tile_rows: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    halo = required_compiled_psf_halo(compiled_rgb)
    output = np.empty_like(array)
    for y0 in range(0, array.shape[0], tile_rows):
        y1 = min(array.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(array.shape[0], y1 + halo)
        rendered = apply_compiled_positive_psf(
            array[source_y0:source_y1], compiled_rgb
        )
        output[y0:y1] = rendered[y0 - source_y0 : y1 - source_y0]
    return output


def channel_psf_from_json(value: Mapping[str, Any]) -> ChannelPsf:
    components = tuple(
        PositivePsfComponent(weight=float(row["weight"]), sigma_um=float(row["sigma_um"]))
        for row in value["components"]
    )
    return ChannelPsf(family=str(value["family"]), components=components)


__all__ = [
    "BUNDLE_SCHEMA",
    "ChannelPsf",
    "CompiledPsfComponent",
    "PositivePsfComponent",
    "apply_compiled_positive_psf",
    "apply_compiled_positive_psf_row_tiled",
    "channel_psf_from_json",
    "compile_channel_psf",
    "compiled_channel_response",
    "required_compiled_psf_halo",
]
