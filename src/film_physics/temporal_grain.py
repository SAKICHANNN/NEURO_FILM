"""Counter-addressed temporally independent grain innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    coordinate_counter_u64,
)
from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.structure_compiler import counter_normal_region
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt
from src.film_physics.thomas_image_formation import (
    CHANNELS,
    ThomasImageFormationContext,
    ThomasImageFormationResult,
    render_typed_thomas_image_formation,
)


@dataclass(frozen=True)
class TemporalTypedThomasFrame:
    result: ThomasImageFormationResult
    receipts: tuple
    context: ThomasImageFormationContext


def build_temporal_reference_exposure(
    shape: tuple[int, int],
    prior: ManufacturerCharacteristicPrior,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
    pixel_pitch_um: float,
) -> tuple[PhysicalDomainArray, np.ndarray]:
    """Build the deterministic three-layer exposure ramp used by temporal audits."""
    y = np.linspace(0.0, 1.0, shape[0], dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)[None, :]
    fractions = np.stack(
        (
            np.broadcast_to(x, shape),
            np.broadcast_to(y, shape),
            0.5 * (np.broadcast_to(x, shape) + np.broadcast_to(y, shape)),
        ),
        axis=-1,
    )
    fractions = minimum_fraction + (maximum_fraction - minimum_fraction) * fractions
    relative_log = np.empty_like(fractions)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        relative_log[..., index] = lower + fractions[..., index] * (upper - lower)
    return (
        PhysicalDomainArray(
            np.power(10.0, relative_log),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            CHANNELS,
            PhysicalScale(pixel_pitch_um),
        ),
        relative_log,
    )


def baseline_scan_from_log_exposure(
    prior: ManufacturerCharacteristicPrior, relative_log: np.ndarray
) -> np.ndarray:
    """Return no-grain identity-scanner transmittance for one typed exposure field."""
    density = np.empty_like(relative_log)
    for index, curve in enumerate(prior.curves):
        density[..., index] = curve.apply(relative_log[..., index])
    return np.power(10.0, -density)


def temporal_grain_innovation_region(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    layer: int,
    full_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return one frame/layer innovation region with exact tile semantics."""
    stream_seed = coordinate_counter_u64(
        profile_sha256=profile_sha256,
        seed=seed,
        frame=frame,
        x=0,
        y=0,
        layer=layer,
    )
    with np.errstate(over="ignore"):
        return counter_normal_region(
            full_shape,
            origin_yx=origin_yx,
            shape=shape,
            seed=stream_seed,
        )


def temporal_grain_innovation_frame(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    full_shape: tuple[int, int],
    layer_count: int,
) -> np.ndarray:
    """Return an HWC stack of independent frame/layer innovations."""
    if not isinstance(layer_count, int) or layer_count < 1:
        raise ValueError("layer count must be positive")
    return np.stack(
        [
            temporal_grain_innovation_region(
                profile_sha256=profile_sha256,
                seed=seed,
                frame=frame,
                layer=layer,
                full_shape=full_shape,
                origin_yx=(0, 0),
                shape=full_shape,
            )
            for layer in range(layer_count)
        ],
        axis=-1,
    )


def temporal_grain_realization_seeds(
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    layer_count: int,
) -> tuple[int, ...]:
    """Derive frame/layer-addressed seeds for existing physical field engines."""
    if not isinstance(layer_count, int) or layer_count < 1:
        raise ValueError("layer count must be positive")
    return tuple(
        coordinate_counter_u64(
            profile_sha256=profile_sha256,
            seed=seed,
            frame=frame,
            x=0,
            y=0,
            layer=layer,
        )
        for layer in range(layer_count)
    )


def render_temporal_typed_thomas_frame(
    exposure: PhysicalDomainArray,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    *,
    profile_sha256: str,
    seed: int,
    frame: int,
    scanner_profile_id: str,
    canonical_receipt_row_block_height: int,
) -> TemporalTypedThomasFrame:
    """Render one frame through the retained typed Thomas image-formation path."""
    shape = exposure.values.shape[:2]
    seeds = temporal_grain_realization_seeds(
        profile_sha256=profile_sha256,
        seed=seed,
        frame=frame,
        layer_count=3,
    )
    receipts = tuple(
        build_thomas_dc_receipt(
            shape,
            profile_id=profile.spatial_profile_id,
            particle_sigma_pixels=profile.particle_sigma_samples,
            cluster_sigma_pixels=profile.cluster_sigma_samples,
            mean_offspring=profile.mean_offspring,
            component_seeds=profile.component_seeds,
            realization_seed=realization_seed,
            truncate=profile.truncate,
            canonical_row_block_height=canonical_receipt_row_block_height,
        )
        for realization_seed in seeds
    )
    context = ThomasImageFormationContext(
        profile_id=profile.identity(),
        prior_id=prior.identity(),
        scanner_profile_id=scanner_profile_id,
        full_shape=shape,
        receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
        layer_realization_seeds=seeds,
    )
    result = render_typed_thomas_image_formation(exposure, profile, prior, receipts, context)
    return TemporalTypedThomasFrame(result=result, receipts=receipts, context=context)
