"""Explicit physical-domain contracts and reference simulation primitives."""

from .contracts import (
    CANONICAL_DOMAIN_ORDER,
    DOMAIN_UNITS,
    PROFILE_BUNDLE_SCHEMA,
    QUALITY_TIERS,
    ComponentBinding,
    FilmProfileBundle,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    QualityTier,
    QualityTierSpec,
    coordinate_counter_u64,
    density_to_transmittance,
    scene_exposure_from_working_image,
    transmittance_to_density,
)

__all__ = [
    "CANONICAL_DOMAIN_ORDER",
    "DOMAIN_UNITS",
    "PROFILE_BUNDLE_SCHEMA",
    "QUALITY_TIERS",
    "ComponentBinding",
    "FilmProfileBundle",
    "PhysicalDomain",
    "PhysicalDomainArray",
    "PhysicalScale",
    "PhysicalUnit",
    "QualityTier",
    "QualityTierSpec",
    "coordinate_counter_u64",
    "density_to_transmittance",
    "scene_exposure_from_working_image",
    "transmittance_to_density",
]
