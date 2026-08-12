"""Compose retained exposure-domain spatial operators before native Thomas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .backing_return import backing_return_profile_from_contract
from .compiled_backing_return import (
    CompiledBackingReturnProfile,
    apply_compiled_backing_return,
    compile_backing_return_profile,
)
from .compiled_scatter import (
    CompiledScatterProfile,
    apply_compiled_scatter,
    compile_scatter_profile,
)
from .contracts import PhysicalDomain, PhysicalDomainArray
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


__all__ = [
    "NativeThomasSpatialChain",
    "apply_native_thomas_spatial_chain",
    "compile_native_thomas_spatial_chain",
]
