"""Typed generic B&W development to metallic-silver image structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roll2film.sensitometry import RGBSensitometryOperator

from .contracts import PhysicalDomain, PhysicalDomainArray
from .developed_structure import (
    DevelopedStructureContext,
    build_bw_silver_context,
    render_developed_structure,
    render_developed_structure_region,
)
from .exposure_development import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    develop_layer_exposure,
)


@dataclass(frozen=True)
class BwSilverChainResult:
    mean_silver_density: np.ndarray
    transmittance: PhysicalDomainArray
    context: DevelopedStructureContext
    interpretation_contract: DevelopmentInterpretationContract


def build_typed_bw_silver_chain(
    layer_exposure: PhysicalDomainArray,
    operator: RGBSensitometryOperator,
    interpretation: DevelopmentInterpretationContract,
    *,
    radius_um: float,
    output_zoom: int,
    output_pixel_pitch_um: float,
    monte_carlo_samples: int,
    seed: int,
) -> BwSilverChainResult:
    if interpretation.emulsion_family is not EmulsionFamily.BLACK_AND_WHITE:
        raise ValueError("metallic-silver chain requires a B&W interpretation")
    developed = develop_layer_exposure(layer_exposure, operator, interpretation).density
    neutral_density = np.mean(developed.values.astype(np.float64), axis=-1)
    context = build_bw_silver_context(
        neutral_density,
        radius_um=radius_um,
        output_zoom=output_zoom,
        output_pixel_pitch_um=output_pixel_pitch_um,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    transmittance = render_developed_structure(context)
    transmittance.require(PhysicalDomain.TRANSMITTANCE)
    return BwSilverChainResult(neutral_density, transmittance, context, interpretation)


def render_bw_silver_chain_region(
    result: BwSilverChainResult,
    *,
    output_origin_yx: tuple[int, int],
    output_shape: tuple[int, int],
) -> PhysicalDomainArray:
    return render_developed_structure_region(
        result.context,
        output_origin_yx=output_origin_yx,
        output_shape=output_shape,
    )


__all__ = [
    "BwSilverChainResult",
    "build_typed_bw_silver_chain",
    "render_bw_silver_chain_region",
]
