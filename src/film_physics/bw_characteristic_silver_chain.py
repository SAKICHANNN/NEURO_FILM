"""Research-only B&W characteristic curve to metallic-silver structure chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bw_characteristic_surface import BWCharacteristicSurface
from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalScale, PhysicalUnit
from .developed_structure import (
    DevelopedStructureContext,
    build_bw_silver_context,
    render_developed_structure,
    render_developed_structure_region,
)


@dataclass(frozen=True)
class BWCharacteristicSilverChainResult:
    development_time_minutes: float
    developed_density: PhysicalDomainArray
    transmittance: PhysicalDomainArray
    context: DevelopedStructureContext


def build_bw_characteristic_silver_chain(
    relative_log_exposure: np.ndarray,
    surface: BWCharacteristicSurface,
    *,
    development_time_minutes: float,
    radius_um: float,
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
    seed: int,
) -> BWCharacteristicSilverChainResult:
    """Develop a 2-D source-graph coordinate into metallic-silver structure.

    The input intentionally is not a scene- or layer-exposure physical array;
    the source graph supplies only relative log-exposure coordinates.
    """

    if not isinstance(surface, BWCharacteristicSurface):
        raise TypeError("surface must be a BWCharacteristicSurface")
    exposure = np.asarray(relative_log_exposure)
    if exposure.ndim != 2 or exposure.size == 0:
        raise ValueError("relative graph log exposure must be a non-empty 2-D field")
    density = surface.density(development_time_minutes, exposure)
    density_rgb = np.repeat(density[..., None], 3, axis=-1)
    scale = PhysicalScale(float(output_zoom) * float(output_pixel_pitch_um))
    developed = PhysicalDomainArray(
        density_rgb,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        ("neutral", "neutral", "neutral"),
        scale,
    )
    context = build_bw_silver_context(
        density,
        radius_um=radius_um,
        output_zoom=output_zoom,
        output_pixel_pitch_um=output_pixel_pitch_um,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    transmittance = render_developed_structure(context)
    transmittance.require(PhysicalDomain.TRANSMITTANCE)
    return BWCharacteristicSilverChainResult(
        float(development_time_minutes), developed, transmittance, context
    )


def render_bw_characteristic_silver_chain_region(
    result: BWCharacteristicSilverChainResult,
    *,
    output_origin_yx: tuple[int, int],
    output_shape: tuple[int, int],
) -> PhysicalDomainArray:
    if not isinstance(result, BWCharacteristicSilverChainResult):
        raise TypeError("result must be a BWCharacteristicSilverChainResult")
    return render_developed_structure_region(
        result.context,
        output_origin_yx=output_origin_yx,
        output_shape=output_shape,
    )


__all__ = [
    "BWCharacteristicSilverChainResult",
    "build_bw_characteristic_silver_chain",
    "render_bw_characteristic_silver_chain_region",
]
