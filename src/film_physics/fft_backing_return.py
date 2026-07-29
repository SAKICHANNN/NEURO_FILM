"""Float32 FFT challenger for the compiled bounded backing-return profile."""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np
from scipy.signal import fftconvolve

from .compiled_backing_return import CompiledBackingReturnProfile
from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit


FFT_RETURN_ROUNDOFF_FLOOR = -2e-6


def _apply_fft_backing_return_values(
    values: np.ndarray,
    profile: CompiledBackingReturnProfile,
) -> tuple[np.ndarray, float]:
    output = np.array(values, dtype=np.float32, copy=True, order="C")
    minimum_raw_returned = 0.0
    for component in profile.kernels:
        kernel_2d = np.multiply.outer(
            component.weights, component.weights
        ).astype(np.float32)
        kernel_2d /= np.sum(kernel_2d, dtype=np.float32)
        for source_channel in range(3):
            source_weights = component.return_weights[:, source_channel]
            if not np.any(source_weights):
                continue
            returned = fftconvolve(
                values[..., source_channel], kernel_2d, mode="same"
            ).astype(np.float32, copy=False)
            minimum_raw_returned = min(
                minimum_raw_returned, float(np.min(returned))
            )
            if minimum_raw_returned < FFT_RETURN_ROUNDOFF_FLOOR:
                raise RuntimeError("FFT backing return exceeded roundoff floor")
            returned = np.maximum(returned, np.float32(0.0))
            for target_layer in range(3):
                weight = source_weights[target_layer]
                if weight != 0.0:
                    output[..., target_layer] += weight * returned
    if np.any(output < values):
        raise RuntimeError("FFT backing return subtracted direct exposure")
    return output, minimum_raw_returned


def apply_fft_backing_return(
    exposure: PhysicalDomainArray,
    profile: CompiledBackingReturnProfile,
) -> PhysicalDomainArray:
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float32:
        raise TypeError("FFT backing return requires float32 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("FFT backing return requires an HxWx3 image")
    if exposure.scale is None or exposure.scale.pixel_pitch_um != profile.pixel_pitch_um:
        raise ValueError("FFT backing-return pixel scale does not match profile")
    output, _ = _apply_fft_backing_return_values(exposure.values, profile)
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )


def apply_fft_backing_return_row_tiled(
    exposure: PhysicalDomainArray,
    profile: CompiledBackingReturnProfile,
    *,
    tile_rows: int,
) -> PhysicalDomainArray:
    """Finite-support row execution with tolerance, not bit-exact, FFT parity."""

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
        rendered = apply_fft_backing_return(window, profile).values
        crop_y0 = y0 - source_y0
        output[y0:y1] = rendered[crop_y0 : crop_y0 + (y1 - y0)]
    return PhysicalDomainArray(
        output,
        exposure.domain,
        exposure.unit,
        exposure.channels,
        exposure.scale,
    )


def iter_fft_backing_return_row_cores(
    exposure: PhysicalDomainArray,
    profile: CompiledBackingReturnProfile,
    *,
    tile_rows: int,
    order: str = "forward",
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield owned row cores without allocating a full output image."""

    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float32:
        raise TypeError("streamed FFT backing return requires float32 exposure")
    if exposure.scale is None or exposure.scale.pixel_pitch_um != profile.pixel_pitch_um:
        raise ValueError("streamed FFT backing-return pixel scale does not match profile")
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    height = exposure.values.shape[0]
    ranges = [
        (y0, min(height, y0 + tile_rows))
        for y0 in range(0, height, tile_rows)
    ]
    if order == "reverse":
        ranges.reverse()
    halo = profile.required_halo
    for y0, y1 in ranges:
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        window = PhysicalDomainArray(
            exposure.values[source_y0:source_y1],
            exposure.domain,
            exposure.unit,
            exposure.channels,
            exposure.scale,
        )
        rendered = apply_fft_backing_return(window, profile).values
        crop_y0 = y0 - source_y0
        core = np.array(
            rendered[crop_y0 : crop_y0 + (y1 - y0)],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        yield y0, y1, core
