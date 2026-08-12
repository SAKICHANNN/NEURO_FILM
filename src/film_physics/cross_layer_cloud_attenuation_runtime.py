"""Compiled row runtime for a target-resolution cross-layer cloud approximation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.eval.target_resolution_cloud_lod_v2 import compile_target_resolution_profile

from .cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from .cross_layer_cloud_runtime import (
    estimate_cross_layer_cloud_row_stream_live_bytes,
    iter_optical_density_cross_layer_cloud_rows_v2,
)
from .density_conditioned_structure import DensityConditionedStructureResult


@dataclass(frozen=True)
class CompiledCloudAttenuationProfile:
    base_profile: CrossLayerCloudReferenceProfile
    channel_residual_gain: tuple[float, float, float]
    aperture_factor: int
    base_rate_multiplier: float
    product_enabled: bool = False

    def __post_init__(self) -> None:
        gain = np.asarray(self.channel_residual_gain, dtype=np.float64)
        if (
            not isinstance(self.base_profile, CrossLayerCloudReferenceProfile)
            or self.base_profile.product_enabled
            or self.product_enabled
            or not isinstance(self.aperture_factor, int)
            or self.aperture_factor < 1
            or not np.isfinite(self.base_rate_multiplier)
            or self.base_rate_multiplier <= 0.0
            or gain.shape != (3,)
            or not np.all(np.isfinite(gain))
            or np.any(gain <= 0.0)
            or np.any(gain > 1.0)
        ):
            raise ValueError("invalid compiled cloud attenuation profile")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "neuro_film.compiled_cloud_attenuation_profile.v1",
            "base_profile": self.base_profile.to_payload(),
            "channel_residual_gain": list(self.channel_residual_gain),
            "aperture_factor": self.aperture_factor,
            "base_rate_multiplier": self.base_rate_multiplier,
            "product_enabled": self.product_enabled,
        }

    def identity(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> CompiledCloudAttenuationProfile:
        expected = {
            "schema",
            "base_profile",
            "channel_residual_gain",
            "aperture_factor",
            "base_rate_multiplier",
            "product_enabled",
        }
        if (
            set(payload) != expected
            or payload.get("schema")
            != "neuro_film.compiled_cloud_attenuation_profile.v1"
            or payload.get("product_enabled") is not False
            or not isinstance(payload.get("base_profile"), dict)
        ):
            raise ValueError("unsupported compiled cloud attenuation profile")
        result = cls(
            base_profile=CrossLayerCloudReferenceProfile.from_payload(
                payload["base_profile"]  # type: ignore[arg-type]
            ),
            channel_residual_gain=tuple(payload["channel_residual_gain"]),  # type: ignore[arg-type]
            aperture_factor=payload["aperture_factor"],  # type: ignore[arg-type]
            base_rate_multiplier=payload["base_rate_multiplier"],  # type: ignore[arg-type]
            product_enabled=False,
        )
        if result.to_payload() != payload:
            raise ValueError("compiled cloud attenuation derived identity drift")
        return result


def compile_cloud_attenuation_profile(
    reference: CrossLayerCloudReferenceProfile,
    *,
    aperture_factor: int,
    base_rate_multiplier: float,
    channel_residual_gain: tuple[float, float, float],
) -> CompiledCloudAttenuationProfile:
    return CompiledCloudAttenuationProfile(
        base_profile=compile_target_resolution_profile(
            reference,
            aperture_factor,
            rate_multiplier=base_rate_multiplier,
        ),
        channel_residual_gain=channel_residual_gain,
        aperture_factor=aperture_factor,
        base_rate_multiplier=base_rate_multiplier,
    )


def iter_compiled_cloud_attenuation_rows(
    profile: CompiledCloudAttenuationProfile,
    target_optical_density_cmy: np.ndarray,
    *,
    seed: int,
    row_tile_height: int,
) -> Iterator[tuple[int, DensityConditionedStructureResult]]:
    target = np.asarray(target_optical_density_cmy, dtype=np.float64)
    gain = np.asarray(profile.channel_residual_gain, dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[-1] != 3
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or not isinstance(row_tile_height, int)
        or row_tile_height < 1
    ):
        raise ValueError("invalid compiled cloud attenuation request")
    # Construct the upstream iterator only after validating every local input.
    base_rows = iter_optical_density_cross_layer_cloud_rows_v2(
        profile.base_profile,
        target,
        seed=seed,
        row_tile_height=row_tile_height,
    )
    for y0, base in base_rows:
        expected = np.power(10.0, -target[y0 : y0 + base.density.shape[0]])
        candidate = expected + gain * (base.transmittance.astype(np.float64) - expected)
        if not np.all(np.isfinite(candidate)) or np.any(candidate <= 0.0) or np.any(
            candidate >= 1.0
        ):
            raise ValueError("compiled cloud attenuation left its open-unit domain")
        density = (-np.log10(candidate)).astype(np.float32)
        transmittance = np.power(10.0, -density.astype(np.float64)).astype(np.float32)
        yield y0, DensityConditionedStructureResult(density, transmittance)


def estimate_compiled_cloud_attenuation_live_bytes(
    profile: CompiledCloudAttenuationProfile,
    *,
    full_shape: tuple[int, int],
    row_tile_height: int,
) -> int:
    base = estimate_cross_layer_cloud_row_stream_live_bytes(
        profile.base_profile,
        full_shape=full_shape,
        row_tile_height=row_tile_height,
    )
    rows = min(row_tile_height, full_shape[0])
    # Expected and candidate float64 RGB rows plus returned float32 density/T.
    return base + rows * full_shape[1] * 3 * (8 * 2 + 4 * 2)


__all__ = [
    "CompiledCloudAttenuationProfile",
    "compile_cloud_attenuation_profile",
    "estimate_compiled_cloud_attenuation_live_bytes",
    "iter_compiled_cloud_attenuation_rows",
]
