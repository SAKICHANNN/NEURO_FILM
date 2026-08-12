"""Versioned research profile for cross-layer Gaussian cloud structure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .cross_layer_aperture_lod import gaussian_aperture_effective_kernel
from .cross_layer_compound_poisson import CrossLayerPoissonProfile

SCHEMA = "film_physics.cross_layer_cloud_reference_profile.v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@dataclass(frozen=True)
class CrossLayerCloudReferenceProfile:
    count_profile: CrossLayerPoissonProfile
    gaussian_sigma_pixels_cmy: tuple[float, float, float]
    gaussian_truncate: float
    aperture_factors: tuple[int, ...]
    parent_evidence_sha256: tuple[str, str, str]
    product_enabled: bool = False

    def to_payload(self) -> dict[str, Any]:
        analytic = self.count_profile.analytic_correlation()
        kernels = {
            str(factor): [
                hashlib.sha256(
                    gaussian_aperture_effective_kernel(
                        sigma, factor, truncate=self.gaussian_truncate
                    )
                    .astype("<f8")
                    .tobytes()
                ).hexdigest()
                for sigma in self.gaussian_sigma_pixels_cmy
            ]
            for factor in self.aperture_factors
        }
        return {
            "schema": SCHEMA,
            "marginal_count_rates_cmy": list(self.count_profile.marginal_rates_cmy),
            "shared_all_rate": self.count_profile.shared_all_rate,
            "shared_pair_rates_cm_cy_my": list(
                self.count_profile.shared_pair_rates_cm_cy_my
            ),
            "mark_optical_density_cmy": list(
                self.count_profile.mark_optical_density_cmy
            ),
            "gaussian_sigma_pixels_cmy": list(self.gaussian_sigma_pixels_cmy),
            "gaussian_truncate": self.gaussian_truncate,
            "aperture_factors": list(self.aperture_factors),
            "component_seed_stride": self.count_profile.component_seed_stride,
            "analytic_count_correlation": analytic.tolist(),
            "aperture_kernel_sha256_cmy": kernels,
            "parent_evidence_sha256": list(self.parent_evidence_sha256),
            "product_enabled": self.product_enabled,
            "claim_ceiling": "synthetic-research-reference-not-calibrated-not-product",
        }

    def identity(self) -> str:
        return hashlib.sha256(_canonical(self.to_payload())).hexdigest()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CrossLayerCloudReferenceProfile:
        expected_keys = {
            "schema",
            "marginal_count_rates_cmy",
            "shared_all_rate",
            "shared_pair_rates_cm_cy_my",
            "mark_optical_density_cmy",
            "gaussian_sigma_pixels_cmy",
            "gaussian_truncate",
            "aperture_factors",
            "component_seed_stride",
            "analytic_count_correlation",
            "aperture_kernel_sha256_cmy",
            "parent_evidence_sha256",
            "product_enabled",
            "claim_ceiling",
        }
        if (
            set(payload) != expected_keys
            or payload.get("schema") != SCHEMA
            or payload.get("product_enabled") is not False
            or payload.get("claim_ceiling")
            != "synthetic-research-reference-not-calibrated-not-product"
        ):
            raise ValueError("unsupported cross-layer cloud profile")
        count = CrossLayerPoissonProfile(
            tuple(payload["marginal_count_rates_cmy"]),
            payload["shared_all_rate"],
            tuple(payload["shared_pair_rates_cm_cy_my"]),
            tuple(payload["mark_optical_density_cmy"]),
            0,
            payload["component_seed_stride"],
        )
        result = cls(
            count,
            tuple(payload["gaussian_sigma_pixels_cmy"]),
            payload["gaussian_truncate"],
            tuple(payload["aperture_factors"]),
            tuple(payload["parent_evidence_sha256"]),
            False,
        )
        if result.to_payload() != payload:
            raise ValueError("cross-layer cloud derived identity drift")
        return result


__all__ = ["SCHEMA", "CrossLayerCloudReferenceProfile"]
