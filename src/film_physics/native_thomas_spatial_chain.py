"""Compose retained exposure-domain spatial operators before native Thomas."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backing_return import backing_return_profile_from_contract
from .compiled_backing_return import (
    CompiledBackingReturnProfile,
    apply_compiled_backing_return,
    apply_compiled_backing_return_row_tiled,
    compile_backing_return_profile,
)
from .compiled_scatter import (
    CompiledScatterProfile,
    apply_compiled_scatter,
    apply_compiled_scatter_row_tiled,
    compile_scatter_profile,
)
from .contracts import PhysicalDomain, PhysicalDomainArray
from .fft_backing_return import apply_fft_backing_return
from .reference_scatter import profile_from_contract


@dataclass(frozen=True)
class NativeThomasSpatialChain:
    """Fixed forward-scatter then positive backing-return exposure chain."""

    forward_scatter: CompiledScatterProfile
    backing_return: CompiledBackingReturnProfile

    def __post_init__(self) -> None:
        if self.forward_scatter.pixel_pitch_um != self.backing_return.pixel_pitch_um:
            raise ValueError("native Thomas spatial-chain physical scale mismatch")

    @property
    def pixel_pitch_um(self) -> float:
        return self.forward_scatter.pixel_pitch_um


def compile_native_thomas_spatial_chain(
    p1_contract: dict[str, Any], p3d_contract: dict[str, Any]
) -> NativeThomasSpatialChain:
    """Compile only P1 forward scatter plus the distinct P3D additive return."""
    forward_contract = json.loads(json.dumps(p1_contract))
    forward_contract["components"] = [
        component
        for component in forward_contract.get("components", [])
        if component.get("component_id") == "forward-scatter"
    ]
    if len(forward_contract["components"]) != 1:
        raise ValueError("P1 forward-scatter component identity drift")
    forward = compile_scatter_profile(profile_from_contract(forward_contract))
    backing = compile_backing_return_profile(
        backing_return_profile_from_contract(p3d_contract)
    )
    return NativeThomasSpatialChain(forward, backing)


def apply_native_thomas_spatial_chain(
    exposure: PhysicalDomainArray, chain: NativeThomasSpatialChain
) -> PhysicalDomainArray:
    """Apply the frozen physical order without clipping or domain relabeling."""
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != (
        "red-sensitive",
        "green-sensitive",
        "blue-sensitive",
    ):
        raise ValueError("native Thomas spatial chain requires sensitive-layer order")
    forward = apply_compiled_scatter(exposure, chain.forward_scatter)
    return apply_compiled_backing_return(forward, chain.backing_return)


def apply_native_thomas_spatial_chain_fft(
    exposure: PhysicalDomainArray, chain: NativeThomasSpatialChain
) -> PhysicalDomainArray:
    """Apply the same physical order with the retained FFT backing challenger."""
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != (
        "red-sensitive",
        "green-sensitive",
        "blue-sensitive",
    ):
        raise ValueError("native Thomas spatial chain requires sensitive-layer order")
    forward = apply_compiled_scatter(exposure, chain.forward_scatter)
    return apply_fft_backing_return(forward, chain.backing_return)


def apply_native_thomas_spatial_chain_row_tiled(
    exposure: PhysicalDomainArray,
    chain: NativeThomasSpatialChain,
    *,
    tile_rows: int,
) -> PhysicalDomainArray:
    """Execute the same physical order with exact full-frame halo windows."""
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != (
        "red-sensitive",
        "green-sensitive",
        "blue-sensitive",
    ):
        raise ValueError("native Thomas spatial chain requires sensitive-layer order")
    forward = apply_compiled_scatter_row_tiled(
        exposure, chain.forward_scatter, tile_rows=tile_rows
    )
    return apply_compiled_backing_return_row_tiled(
        forward, chain.backing_return, tile_rows=tile_rows
    )


def iter_native_thomas_spatial_fft_row_cores(
    exposure: PhysicalDomainArray,
    chain: NativeThomasSpatialChain,
    *,
    tile_rows: int,
    order: str = "forward",
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield combined-halo forward-scatter plus FFT-backing row cores."""
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != (
        "red-sensitive",
        "green-sensitive",
        "blue-sensitive",
    ):
        raise ValueError("native Thomas spatial chain requires sensitive-layer order")
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    height = exposure.values.shape[0]
    ranges = [(y0, min(height, y0 + tile_rows)) for y0 in range(0, height, tile_rows)]
    if order == "reverse":
        ranges.reverse()
    scatter_halo = chain.forward_scatter.required_halo
    backing_halo = chain.backing_return.required_halo
    combined_halo = scatter_halo + backing_halo
    for y0, y1 in ranges:
        source_y0 = max(0, y0 - combined_halo)
        source_y1 = min(height, y1 + combined_halo)
        source_window = PhysicalDomainArray(
            exposure.values[source_y0:source_y1],
            exposure.domain,
            exposure.unit,
            exposure.channels,
            exposure.scale,
        )
        scattered = apply_compiled_scatter(source_window, chain.forward_scatter)
        backing_y0 = max(0, y0 - backing_halo)
        backing_y1 = min(height, y1 + backing_halo)
        backing_window = PhysicalDomainArray(
            scattered.values[
                backing_y0 - source_y0 : backing_y1 - source_y0
            ],
            exposure.domain,
            exposure.unit,
            exposure.channels,
            exposure.scale,
        )
        returned = apply_fft_backing_return(backing_window, chain.backing_return)
        crop_y0 = y0 - backing_y0
        core = returned.values[crop_y0 : crop_y0 + (y1 - y0)].copy(order="C")
        yield y0, y1, core


__all__ = [
    "NativeThomasSpatialChain",
    "apply_native_thomas_spatial_chain",
    "apply_native_thomas_spatial_chain_fft",
    "apply_native_thomas_spatial_chain_row_tiled",
    "compile_native_thomas_spatial_chain",
    "iter_native_thomas_spatial_fft_row_cores",
]
