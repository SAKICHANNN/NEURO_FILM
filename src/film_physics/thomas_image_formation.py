"""Typed image formation for the retained density-conditioned Thomas profile.

The P4BS spectrum was measured from a developed scan and therefore already
contains the source scanner's spatial convolution.  This module deliberately
uses an identity spectral observer only; adding scanner MTF, flare, or noise
would double count that measurement path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    density_to_transmittance,
)
from .density_conditioned_thomas import DensityConditionedThomasProfile
from .manufacturer_characteristic import ManufacturerCharacteristicPrior
from .scanner import ScannerProfile, apply_scanner_profile
from .thomas_dc_projection import ThomasDcReceipt

THOMAS_IMAGE_FORMATION_SCHEMA = "neuro_film.typed_thomas_image_formation_context.v1"
CHANNELS = ("red", "green", "blue")


def effective_spectrum_identity_scanner(profile_id: str) -> ScannerProfile:
    """Return the explicit no-extra-spatial-processing scan observer."""
    return ScannerProfile(
        profile_id=profile_id,
        illuminant_rgb=(1.0, 1.0, 1.0),
        spectral_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        local_flare_fraction=0.0,
        global_flare_fraction=0.0,
        flare_sigma_um=0.0,
        dmax_density_rgb=None,
        mtf_sigma_um_rgb=(0.0, 0.0, 0.0),
        shot_noise_variance_scale=0.0,
        read_noise_variance=0.0,
        seed=0,
    )


@dataclass(frozen=True)
class ThomasImageFormationContext:
    profile_id: str
    prior_id: str
    scanner_profile_id: str
    full_shape: tuple[int, int]
    receipt_ids: tuple[str, str, str]
    layer_realization_seeds: tuple[int, int, int]
    scanner_convolution_embedded: bool = True
    scanner_stages: tuple[str, ...] = ("spectral",)

    def __post_init__(self) -> None:
        if not self.profile_id or not self.prior_id or not self.scanner_profile_id:
            raise ValueError("image-formation identities must be non-empty")
        if (
            len(self.full_shape) != 2
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
            or self.scanner_convolution_embedded is not True
            or self.scanner_stages != ("spectral",)
        ):
            raise ValueError("invalid typed Thomas image-formation context")

    @property
    def context_id(self) -> str:
        payload = {
            "schema": THOMAS_IMAGE_FORMATION_SCHEMA,
            "profile_id": self.profile_id,
            "prior_id": self.prior_id,
            "scanner_profile_id": self.scanner_profile_id,
            "full_shape": list(self.full_shape),
            "receipt_ids": list(self.receipt_ids),
            "layer_realization_seeds": list(self.layer_realization_seeds),
            "scanner_convolution_embedded": self.scanner_convolution_embedded,
            "scanner_stages": list(self.scanner_stages),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class ThomasImageFormationResult:
    developed_density: PhysicalDomainArray
    transmittance: PhysicalDomainArray
    scan_linear: PhysicalDomainArray
    context_id: str


def render_typed_thomas_image_formation_region(
    layer_exposure: PhysicalDomainArray,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    receipts: tuple[ThomasDcReceipt, ThomasDcReceipt, ThomasDcReceipt],
    context: ThomasImageFormationContext,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
) -> ThomasImageFormationResult:
    """Render one coordinate-bound region through density and scan-linear."""
    layer_exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if layer_exposure.values.dtype != np.float64:
        raise TypeError("P4BY reference execution requires float64 layer exposure")
    if layer_exposure.scale is None or (
        layer_exposure.scale.pixel_pitch_um != profile.sample_pitch_micrometres
    ):
        raise ValueError("layer exposure sampling scale does not match profile")
    if layer_exposure.channels != CHANNELS:
        raise ValueError("layer exposure channels must be red, green, blue")
    if layer_exposure.values.shape[:2] != context.full_shape:
        raise ValueError("layer exposure shape does not match context")
    if np.any(layer_exposure.values <= 0.0):
        raise ValueError("layer exposure must be strictly positive before log10")
    if (
        context.profile_id != profile.identity()
        or context.prior_id != prior.identity()
        or tuple(receipt.receipt_id for receipt in receipts) != context.receipt_ids
        or tuple(receipt.realization_seed for receipt in receipts)
        != context.layer_realization_seeds
        or any(receipt.full_shape != context.full_shape for receipt in receipts)
    ):
        raise ValueError("image-formation context does not bind inputs")
    y0, x0 = origin_yx
    height, width = shape
    if not (
        0 <= y0 < y0 + height <= context.full_shape[0]
        and 0 <= x0 < x0 + width <= context.full_shape[1]
    ):
        raise ValueError("image-formation region is outside full field")

    relative_log_exposure = np.log10(layer_exposure.values)
    density_values = np.empty((height, width, 3), dtype=np.float64)
    for index, channel in enumerate(CHANNELS):
        density_values[..., index] = (
            profile.render_nonstationary_developed_density_region(
                receipts[index],
                prior,
                channel=channel,
                full_relative_log_exposure=relative_log_exposure[..., index],
                origin_yx=origin_yx,
                shape=shape,
            )
        )
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
        raise RuntimeError("identity scanner changed embedded-spectrum transmittance")
    scan = PhysicalDomainArray(
        scan_values,
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        CHANNELS,
        layer_exposure.scale,
    )
    return ThomasImageFormationResult(
        developed_density=density,
        transmittance=transmittance,
        scan_linear=scan,
        context_id=context.context_id,
    )


def render_typed_thomas_image_formation(
    layer_exposure: PhysicalDomainArray,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    receipts: tuple[ThomasDcReceipt, ThomasDcReceipt, ThomasDcReceipt],
    context: ThomasImageFormationContext,
) -> ThomasImageFormationResult:
    return render_typed_thomas_image_formation_region(
        layer_exposure,
        profile,
        prior,
        receipts,
        context,
        origin_yx=(0, 0),
        shape=context.full_shape,
    )


__all__ = [
    "THOMAS_IMAGE_FORMATION_SCHEMA",
    "ThomasImageFormationContext",
    "ThomasImageFormationResult",
    "effective_spectrum_identity_scanner",
    "render_typed_thomas_image_formation",
    "render_typed_thomas_image_formation_region",
]
