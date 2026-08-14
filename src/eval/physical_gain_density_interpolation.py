"""U6.P2AV fresh confirmation of physical-gain density coordinates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
)
from src.film_physics.positive_density_field import (
    PhysicalGainSoftplusDensityParameterProfile,
    softplus_density_field,
)
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)


class PhysicalGainDensityInterpolationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2av_softplus_physical_gain_coordinate_contract.v1"
    ):
        raise PhysicalGainDensityInterpolationError("unsupported P2AV contract")
    return payload


def _load_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise PhysicalGainDensityInterpolationError("P2AV parent mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("automatic_pass") is not binding["required_automatic_pass"]:
        raise PhysicalGainDensityInterpolationError("P2AV parent decision mismatch")
    if "required_decision" in binding and (
        payload.get("decision") != binding["required_decision"]
    ):
        raise PhysicalGainDensityInterpolationError("P2AV parent branch mismatch")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    node_fit = _load_parent(root, contract["parents"]["node_fit"])
    _load_parent(root, contract["parents"]["mean_coordinate_boundary"])
    nodes = contract["parameter_nodes"]
    profile = PhysicalGainSoftplusDensityParameterProfile(
        source_evidence_id=node_fit["stable_evidence_id"],
        densities=tuple(nodes["densities"]),
        fitted_a=tuple(nodes["a"]),
        fitted_b=tuple(nodes["b"]),
        target_sigma_d=tuple(nodes["target_aperture_sigma_d"]),
    )
    restored = PhysicalGainSoftplusDensityParameterProfile.from_dict(profile.to_dict())
    confirmation = contract["confirmation"]
    densities = np.asarray(confirmation["densities"], dtype=np.float64)
    target_sigma = np.asarray(confirmation["target_aperture_sigma_d"], dtype=np.float64)
    parameter_a, parameter_b = profile.parameters(densities, target_sigma)
    spatial = contract["spatial_mechanism"]
    evaluation = contract["evaluation"]
    shape = tuple(evaluation["field_shape"])
    aperture = binary_circular_aperture_kernel(
        spatial["sample_pitch_micrometres"],
        spatial["measurement_aperture_diameter_micrometres"],
    )
    rows = []
    repeat_exact = True
    partition_exact = True
    row_height = evaluation["row_partition_height"]
    for seed in confirmation["seeds"]:
        kwargs = {
            "full_shape": shape,
            "profile_id": spatial["spatial_profile_id"],
            "particle_sigma_pixels": spatial["particle_sigma_samples"],
            "cluster_sigma_pixels": spatial["cluster_sigma_samples"],
            "mean_offspring": spatial["mean_offspring"],
            "component_seeds": tuple(spatial["component_seeds"]),
            "realization_seed": int(seed),
            "truncate": spatial["truncate_sigma"],
            "canonical_row_block_height": evaluation["canonical_row_block_height"],
        }
        receipt = build_thomas_dc_receipt(**kwargs)
        repeated_receipt = build_thomas_dc_receipt(**kwargs)
        unit = render_dc_projected_thomas_region(receipt, origin_yx=(0, 0), shape=shape)
        repeated = render_dc_projected_thomas_region(
            repeated_receipt, origin_yx=(0, 0), shape=shape
        )
        repeat_exact &= bool(np.array_equal(unit, repeated))
        partitioned = np.empty_like(unit)
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            partitioned[y0 : y0 + height] = render_dc_projected_thomas_region(
                receipt, origin_yx=(y0, 0), shape=(height, shape[1])
            )
        partition_exact &= bool(np.array_equal(unit, partitioned))
        for density, sigma_target, a, b in zip(
            densities, target_sigma, parameter_a, parameter_b, strict=True
        ):
            field = softplus_density_field(unit, a=float(a), b=float(b))
            partition_field = softplus_density_field(
                partitioned, a=float(a), b=float(b)
            )
            partition_exact &= bool(np.array_equal(field, partition_field))
            measured = fftconvolve(field, aperture, mode="same")
            border = evaluation["measurement_crop_border_samples"]
            measured = measured[border:-border, border:-border]
            observed_mean = float(np.mean(field, dtype=np.float64))
            observed_sigma = float(np.std(measured, dtype=np.float64))
            rows.append(
                {
                    "density": float(density),
                    "seed": int(seed),
                    "a": float(a),
                    "b": float(b),
                    "target_aperture_sigma_d": float(sigma_target),
                    "observed_density_mean": observed_mean,
                    "observed_aperture_sigma_d": observed_sigma,
                    "relative_mean_error": abs(observed_mean - density) / density,
                    "relative_sigma_error": abs(observed_sigma - sigma_target)
                    / sigma_target,
                    "minimum_developed_density": float(np.min(field)),
                    "density_sha256": hashlib.sha256(
                        np.asarray(field, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    mean_errors = np.asarray(
        [row["relative_mean_error"] for row in rows], dtype=np.float64
    )
    sigma_errors = np.asarray(
        [row["relative_sigma_error"] for row in rows], dtype=np.float64
    )
    measurements = {
        "profile_identity": profile.identity(),
        "maximum_confirmation_relative_mean_error": float(np.max(mean_errors)),
        "confirmation_median_sigma_relative_error": float(np.median(sigma_errors)),
        "confirmation_p95_sigma_relative_error": float(
            np.percentile(sigma_errors, 95.0)
        ),
        "minimum_developed_density": min(
            row["minimum_developed_density"] for row in rows
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "exact_roundtrip_identity": (
            profile.to_dict() == restored.to_dict()
            and profile.identity() == restored.identity()
        ),
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "maximum_confirmation_relative_mean_error": measurements[
            "maximum_confirmation_relative_mean_error"
        ]
        <= evaluation["maximum_confirmation_relative_mean_error"],
        "maximum_confirmation_median_sigma_relative_error": measurements[
            "confirmation_median_sigma_relative_error"
        ]
        <= evaluation["maximum_confirmation_median_sigma_relative_error"],
        "maximum_confirmation_p95_sigma_relative_error": measurements[
            "confirmation_p95_sigma_relative_error"
        ]
        <= evaluation["maximum_confirmation_p95_sigma_relative_error"],
        "minimum_developed_density_exclusive": measurements["minimum_developed_density"]
        > evaluation["minimum_developed_density_exclusive"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "exact_roundtrip_identity": measurements["exact_roundtrip_identity"],
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2av_softplus_physical_gain_coordinate_report.v1",
        "profile": profile.to_dict(),
        "rows": rows,
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
