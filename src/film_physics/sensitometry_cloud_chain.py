"""Typed generic colour sensitometry to shared dye-cloud execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.roll2film.sensitometry import RGBSensitometryOperator

from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from .cross_layer_cloud_runtime import iter_target_density_cross_layer_cloud_rows
from .density_conditioned_structure import DensityConditionedStructureResult
from .exposure_development import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    develop_layer_exposure,
)


@dataclass(frozen=True)
class SensitometryCloudChainResult:
    mean_developed_density: PhysicalDomainArray
    cloud_density: PhysicalDomainArray
    transmittance: PhysicalDomainArray
    interpretation_contract: DevelopmentInterpretationContract
    cloud_profile_identity: str


def render_typed_sensitometry_cloud_chain(
    layer_exposure: PhysicalDomainArray,
    operator: RGBSensitometryOperator,
    interpretation: DevelopmentInterpretationContract,
    cloud_profile: CrossLayerCloudReferenceProfile,
    *,
    seed: int,
    row_tile_height: int,
) -> SensitometryCloudChainResult:
    """Develop colour layers then render bounded correlated dye-cloud density."""

    if interpretation.emulsion_family is EmulsionFamily.BLACK_AND_WHITE:
        raise ValueError("B&W requires the separate metallic-silver structure branch")
    developed = develop_layer_exposure(layer_exposure, operator, interpretation).density
    maximum = np.asarray(cloud_profile.count_profile.marginal_rates_cmy) * np.asarray(
        cloud_profile.count_profile.mark_optical_density_cmy
    )
    rows: list[DensityConditionedStructureResult] = []
    for _, result in iter_target_density_cross_layer_cloud_rows(
        cloud_profile,
        developed.values,
        maximum_developed_density_cmy=tuple(maximum),
        seed=seed,
        row_tile_height=row_tile_height,
    ):
        rows.append(result)
    density_values = np.concatenate([row.density for row in rows], axis=0)
    transmittance_values = np.concatenate([row.transmittance for row in rows], axis=0)
    density = PhysicalDomainArray(
        density_values,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        developed.channels,
        developed.scale,
    )
    transmittance = PhysicalDomainArray(
        transmittance_values,
        PhysicalDomain.TRANSMITTANCE,
        PhysicalUnit.TRANSMITTANCE,
        developed.channels,
        developed.scale,
    )
    return SensitometryCloudChainResult(
        developed,
        density,
        transmittance,
        interpretation,
        cloud_profile.identity(),
    )


__all__ = ["SensitometryCloudChainResult", "render_typed_sensitometry_cloud_chain"]
