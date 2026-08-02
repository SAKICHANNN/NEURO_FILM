"""Typed density/transmittance execution for bounded cloud occupancy fields."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .bounded_cloud_occupancy import (
    BoundedCloudDcReceipt,
    render_dc_projected_bounded_cloud_region,
)
from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    density_to_transmittance,
)
from .density_conditioned_thomas import DensityConditionedThomasProfile
from .manufacturer_characteristic import ManufacturerCharacteristicPrior
from .scanner import apply_scanner_profile
from .thomas_image_formation import CHANNELS, effective_spectrum_identity_scanner

BOUNDED_CLOUD_IMAGE_FORMATION_SCHEMA = (
    "neuro_film.typed_bounded_cloud_image_formation_context.v1"
)


@dataclass(frozen=True)
class BoundedCloudImageFormationContext:
    density_profile_id: str
    source_spectrum_profile_id: str
    occupancy_profile_id: str
    prior_id: str
    scanner_profile_id: str
    full_shape: tuple[int, int]
    receipt_ids: tuple[str, str, str]
    layer_realization_seeds: tuple[int, int, int]
    occupancy_trials: int
    occupancy_probability: float
    scanner_convolution_embedded: bool = True
    scanner_stages: tuple[str, ...] = ("spectral",)

    def __post_init__(self) -> None:
        identities = (
            self.density_profile_id,
            self.source_spectrum_profile_id,
            self.occupancy_profile_id,
            self.prior_id,
            self.scanner_profile_id,
        )
        if (
            any(not value for value in identities)
            or len(self.full_shape) != 2
            or any(
                not isinstance(value, int) or value <= 0 for value in self.full_shape
            )
            or len(self.receipt_ids) != 3
            or any(not value for value in self.receipt_ids)
            or len(self.layer_realization_seeds) != 3
            or any(
                not isinstance(value, int) or not 0 <= value < 2**64
                for value in self.layer_realization_seeds
            )
            or not isinstance(self.occupancy_trials, int)
            or self.occupancy_trials < 1
            or not math.isfinite(self.occupancy_probability)
            or not 0.0 < self.occupancy_probability < 1.0
            or self.scanner_convolution_embedded is not True
            or self.scanner_stages != ("spectral",)
        ):
            raise ValueError("invalid bounded cloud image-formation context")

    @property
    def context_id(self) -> str:
        payload = {
            "schema": BOUNDED_CLOUD_IMAGE_FORMATION_SCHEMA,
            "density_profile_id": self.density_profile_id,
            "source_spectrum_profile_id": self.source_spectrum_profile_id,
            "occupancy_profile_id": self.occupancy_profile_id,
            "prior_id": self.prior_id,
            "scanner_profile_id": self.scanner_profile_id,
            "full_shape": list(self.full_shape),
            "receipt_ids": list(self.receipt_ids),
            "layer_realization_seeds": list(self.layer_realization_seeds),
            "occupancy_trials": self.occupancy_trials,
            "occupancy_probability": self.occupancy_probability,
            "scanner_convolution_embedded": self.scanner_convolution_embedded,
            "scanner_stages": list(self.scanner_stages),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class BoundedCloudImageFormationResult:
    developed_density: PhysicalDomainArray
    transmittance: PhysicalDomainArray
    scan_linear: PhysicalDomainArray
    context_id: str


def _validate_receipt(
    receipt: BoundedCloudDcReceipt,
    profile: DensityConditionedThomasProfile,
    context: BoundedCloudImageFormationContext,
) -> None:
    if (
        receipt.profile_id != context.occupancy_profile_id
        or receipt.particle_sigma_pixels != profile.particle_sigma_samples
        or receipt.cluster_sigma_pixels != profile.cluster_sigma_samples
        or receipt.mean_offspring != profile.mean_offspring
        or receipt.component_seeds != profile.component_seeds
        or receipt.truncate != profile.truncate
        or receipt.trials != context.occupancy_trials
        or receipt.probability != context.occupancy_probability
        or receipt.full_shape != context.full_shape
    ):
        raise ValueError("bounded cloud receipt does not match density context")


def render_typed_bounded_cloud_image_formation_region(
    layer_exposure: PhysicalDomainArray,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    receipts: tuple[
        BoundedCloudDcReceipt, BoundedCloudDcReceipt, BoundedCloudDcReceipt
    ],
    context: BoundedCloudImageFormationContext,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> BoundedCloudImageFormationResult:
    layer_exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if layer_exposure.values.dtype != np.float64:
        raise TypeError("bounded cloud reference execution requires float64 exposure")
    if layer_exposure.scale is None or (
        layer_exposure.scale.pixel_pitch_um != profile.sample_pitch_micrometres
    ):
        raise ValueError("layer exposure sampling scale does not match profile")
    if (
        layer_exposure.channels != CHANNELS
        or layer_exposure.values.shape[:2] != context.full_shape
        or np.any(layer_exposure.values <= 0.0)
        or context.density_profile_id != profile.identity()
        or context.source_spectrum_profile_id != profile.spatial_profile_id
        or context.prior_id != prior.identity()
        or tuple(receipt.receipt_id for receipt in receipts) != context.receipt_ids
        or tuple(receipt.realization_seed for receipt in receipts)
        != context.layer_realization_seeds
    ):
        raise ValueError("bounded cloud image-formation context does not bind inputs")
    for receipt in receipts:
        _validate_receipt(receipt, profile, context)
    y0, x0 = origin_yx
    height, width = shape
    if not (
        0 <= y0 < y0 + height <= context.full_shape[0]
        and 0 <= x0 < x0 + width <= context.full_shape[1]
    ):
        raise ValueError("bounded cloud image-formation region is outside full field")

    relative_log_exposure = np.log10(layer_exposure.values)
    density_values = np.empty((height, width, 3), dtype=np.float64)
    for index, channel in enumerate(CHANNELS):
        selected = relative_log_exposure[y0 : y0 + height, x0 : x0 + width, index]
        density_mean = prior.curves[index].apply(selected)
        sigma_d = profile.amplitude_profile.evaluate_channel(prior, channel, selected)
        unit = render_dc_projected_bounded_cloud_region(
            receipts[index], origin_yx=origin_yx, shape=shape
        )
        density_values[..., index] = density_mean + sigma_d * unit / math.sqrt(
            profile.measurement_energy()
        )
    if not np.all(np.isfinite(density_values)) or np.any(density_values < 0.0):
        raise RuntimeError("bounded cloud developed density left physical domain")
    density = PhysicalDomainArray(
        density_values,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        CHANNELS,
        layer_exposure.scale,
    )
    transmittance = density_to_transmittance(density)
    scanner = effective_spectrum_identity_scanner(context.scanner_profile_id)
    scan_values = apply_scanner_profile(
        transmittance.values,
        scanner,
        pixel_pitch_um=layer_exposure.scale.pixel_pitch_um,
        stages=context.scanner_stages,
        full_shape=context.full_shape,
        origin_yx=origin_yx,
    )
    if not np.array_equal(scan_values, transmittance.values):
        raise RuntimeError("identity scanner changed bounded cloud transmittance")
    scan = PhysicalDomainArray(
        scan_values,
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        CHANNELS,
        layer_exposure.scale,
    )
    return BoundedCloudImageFormationResult(
        developed_density=density,
        transmittance=transmittance,
        scan_linear=scan,
        context_id=context.context_id,
    )


def render_typed_bounded_cloud_image_formation(
    layer_exposure: PhysicalDomainArray,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    receipts: tuple[
        BoundedCloudDcReceipt, BoundedCloudDcReceipt, BoundedCloudDcReceipt
    ],
    context: BoundedCloudImageFormationContext,
) -> BoundedCloudImageFormationResult:
    return render_typed_bounded_cloud_image_formation_region(
        layer_exposure,
        profile,
        prior,
        receipts,
        context,
        origin_yx=(0, 0),
        shape=context.full_shape,
    )


__all__ = [
    "BOUNDED_CLOUD_IMAGE_FORMATION_SCHEMA",
    "BoundedCloudImageFormationContext",
    "BoundedCloudImageFormationResult",
    "render_typed_bounded_cloud_image_formation",
    "render_typed_bounded_cloud_image_formation_region",
]
