"""Canonical profile bundle for the bounded generic photographic runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.spatial_response import SpatialResponseProfile

PROFILE_SCHEMA = "neuro-film.bounded-photographic-runtime-profile.v1"
DOMAIN_ORDER = [
    "display-linear-relative-srgb",
    "layer-exposure",
    "developed-density",
    "film-transmittance",
    "paired-scanner-mtf",
    "downstream-display-look",
]


def canonical_profile_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid {field} SHA-256")
    return value


@dataclass(frozen=True)
class BoundedPhotographicRuntimeComponents:
    profile: DensityConditionedThomasProfile
    prior: ManufacturerCharacteristicPrior
    correlation_matrix: np.ndarray
    layer_field_seeds: tuple[int, int, int]
    field_seed_stride_per_source: int
    canonical_row_block_height: int
    rank_bins: int
    scanner_profile: SpatialResponseProfile


def compile_bounded_photographic_profile(
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    correlation_matrix: np.ndarray,
    layer_field_seeds: tuple[int, int, int],
    field_seed_stride_per_source: int,
    canonical_row_block_height: int,
    rank_bins: int,
    scanner_mtf_sigma_pixels_rgb: tuple[float, float, float],
    gaussian_truncate: float,
    source_bindings: dict[str, str],
) -> dict[str, Any]:
    body = {
        "schema": PROFILE_SCHEMA,
        "source_bindings": {
            str(name): _sha256(value, field=f"binding {name}")
            for name, value in sorted(source_bindings.items())
        },
        "density_profile": profile.to_dict(),
        "density_profile_id": profile.identity(),
        "manufacturer_prior": prior.to_dict(),
        "manufacturer_prior_id": prior.identity(),
        "execution": {
            "correlation_matrix": np.asarray(
                correlation_matrix, dtype=np.float64
            ).tolist(),
            "layer_field_seeds": list(layer_field_seeds),
            "field_seed_stride_per_source": field_seed_stride_per_source,
            "canonical_row_block_height": canonical_row_block_height,
            "rank_bins": rank_bins,
        },
        "scanner": {
            "mtf_sigma_pixels_rgb": list(scanner_mtf_sigma_pixels_rgb),
            "gaussian_truncate": gaussian_truncate,
            "baseline_and_candidate_share_stage": True,
        },
        "domain_order": DOMAIN_ORDER,
        "claim_level": "generic-physical-inspired-relative-display",
        "calibrated_stock_or_scanner_claimed": False,
    }
    if not body["source_bindings"]:
        raise ValueError("bounded photographic profile bindings are empty")
    result = {
        **body,
        "bundle_sha256": hashlib.sha256(canonical_profile_bytes(body)).hexdigest(),
    }
    validate_bounded_photographic_profile(result)
    return result


def validate_bounded_photographic_profile(payload: dict[str, Any]) -> str:
    expected = {
        "schema",
        "source_bindings",
        "density_profile",
        "density_profile_id",
        "manufacturer_prior",
        "manufacturer_prior_id",
        "execution",
        "scanner",
        "domain_order",
        "claim_level",
        "calibrated_stock_or_scanner_claimed",
        "bundle_sha256",
    }
    if set(payload) != expected or payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError("bounded photographic profile fields drift")
    if (
        payload.get("domain_order") != DOMAIN_ORDER
        or payload.get("claim_level")
        != "generic-physical-inspired-relative-display"
        or payload.get("calibrated_stock_or_scanner_claimed") is not False
    ):
        raise ValueError("bounded photographic profile claim or domain drift")
    bindings = payload["source_bindings"]
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("bounded photographic profile bindings missing")
    for name, value in bindings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("bounded photographic profile binding name invalid")
        _sha256(value, field=f"binding {name}")
    profile = DensityConditionedThomasProfile.from_dict(payload["density_profile"])
    prior = ManufacturerCharacteristicPrior.from_dict(payload["manufacturer_prior"])
    if (
        _sha256(payload["density_profile_id"], field="density profile")
        != profile.identity()
        or _sha256(payload["manufacturer_prior_id"], field="manufacturer prior")
        != prior.identity()
    ):
        raise ValueError("bounded photographic component identity drift")
    execution = payload["execution"]
    if not isinstance(execution, dict) or set(execution) != {
        "correlation_matrix",
        "layer_field_seeds",
        "field_seed_stride_per_source",
        "canonical_row_block_height",
        "rank_bins",
    }:
        raise ValueError("bounded photographic execution fields drift")
    correlation = np.asarray(execution["correlation_matrix"], dtype=np.float64)
    seeds = execution["layer_field_seeds"]
    if (
        correlation.shape != (3, 3)
        or not np.all(np.isfinite(correlation))
        or not np.array_equal(correlation, correlation.T)
        or not np.array_equal(np.diag(correlation), np.ones(3))
        or np.min(np.linalg.eigvalsh(correlation)) <= 0.0
        or not isinstance(seeds, list)
        or len(seeds) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or any(value < 0 or value >= 2**64 for value in seeds)
        or execution["field_seed_stride_per_source"] != 1009
        or execution["canonical_row_block_height"] != 128
        or execution["rank_bins"] != 65536
    ):
        raise ValueError("bounded photographic execution values drift")
    scanner = payload["scanner"]
    if (
        not isinstance(scanner, dict)
        or set(scanner) != {
            "mtf_sigma_pixels_rgb",
            "gaussian_truncate",
            "baseline_and_candidate_share_stage",
        }
        or scanner["mtf_sigma_pixels_rgb"] != [0.7, 0.7, 0.7]
        or scanner["gaussian_truncate"] != 3.0
        or scanner["baseline_and_candidate_share_stage"] is not True
    ):
        raise ValueError("bounded photographic scanner policy drift")
    bundle_sha = _sha256(payload["bundle_sha256"], field="bundle")
    body = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    if hashlib.sha256(canonical_profile_bytes(body)).hexdigest() != bundle_sha:
        raise ValueError("bounded photographic profile identity drift")
    return bundle_sha


def reconstruct_bounded_photographic_profile(
    payload: dict[str, Any],
) -> BoundedPhotographicRuntimeComponents:
    validate_bounded_photographic_profile(payload)
    execution = payload["execution"]
    scanner = payload["scanner"]
    sigma = tuple(float(value) for value in scanner["mtf_sigma_pixels_rgb"])
    return BoundedPhotographicRuntimeComponents(
        profile=DensityConditionedThomasProfile.from_dict(payload["density_profile"]),
        prior=ManufacturerCharacteristicPrior.from_dict(
            payload["manufacturer_prior"]
        ),
        correlation_matrix=np.asarray(
            execution["correlation_matrix"], dtype=np.float64
        ),
        layer_field_seeds=tuple(int(value) for value in execution["layer_field_seeds"]),
        field_seed_stride_per_source=int(execution["field_seed_stride_per_source"]),
        canonical_row_block_height=int(execution["canonical_row_block_height"]),
        rank_bins=int(execution["rank_bins"]),
        scanner_profile=SpatialResponseProfile(
            pixel_pitch_um=1.0,
            forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
            development_adjacency_sigma_um_rgb=(0.0, 0.0, 0.0),
            development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
            dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
            scanner_mtf_sigma_um_rgb=sigma,
            gaussian_truncate=float(scanner["gaussian_truncate"]),
        ),
    )


def load_bounded_photographic_profile(
    path: Path, *, expected_bundle_sha256: str, maximum_bytes: int = 1048576
) -> dict[str, Any]:
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
        or not path.is_file()
        or path.stat().st_size > maximum_bytes
    ):
        raise ValueError("bounded photographic profile file preflight failed")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bounded photographic profile JSON rejected") from exc
    if not isinstance(payload, dict) or canonical_profile_bytes(payload) != raw:
        raise ValueError("bounded photographic profile is not canonical")
    actual = validate_bounded_photographic_profile(payload)
    if actual != _sha256(expected_bundle_sha256, field="expected bundle"):
        raise ValueError("bounded photographic expected identity drift")
    return payload


__all__ = [
    "PROFILE_SCHEMA",
    "BoundedPhotographicRuntimeComponents",
    "canonical_profile_bytes",
    "compile_bounded_photographic_profile",
    "load_bounded_photographic_profile",
    "reconstruct_bounded_photographic_profile",
    "validate_bounded_photographic_profile",
]
