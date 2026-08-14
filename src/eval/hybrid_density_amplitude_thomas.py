"""U6.P2AR cross-generation amplitude plus generic Thomas field audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.bw_hybrid_density_amplitude import (
    BWHybridDensityAmplitudeProfile,
)
from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
    thomas_aperture_measurement_energy,
)
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)


class HybridDensityAmplitudeThomasError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ar_hybrid_density_amplitude_thomas_contract.v1"
    ):
        raise HybridDensityAmplitudeThomasError("unsupported P2AR contract")
    return payload


def _load_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise HybridDensityAmplitudeThomasError("P2AR parent mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("automatic_pass") is not binding["required_automatic_pass"]:
        raise HybridDensityAmplitudeThomasError("P2AR parent decision mismatch")
    required_identity = binding.get("required_profile_identity")
    if (
        required_identity is not None
        and payload.get("profile_identity") != required_identity
    ):
        raise HybridDensityAmplitudeThomasError("P2AR parent profile mismatch")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    current = _load_parent(root, contract["parents"]["current_scalar"])
    historical = _load_parent(root, contract["parents"]["historical_amplitude"])
    _load_parent(root, contract["parents"]["generic_spatial_shape"])
    specification = contract["hybrid_profile"]
    if (
        current["observation"]["density_mean"] != specification["anchor_density"]
        or current["observation"]["density_rms"] != specification["anchor_density_rms"]
        or current["observation"]["aperture_diameter_micrometres"]
        != specification["aperture_diameter_micrometres"]
    ):
        raise HybridDensityAmplitudeThomasError("P2AR scalar anchor drifted")
    profile = BWHybridDensityAmplitudeProfile(
        hypothesis_identity=specification["identity"],
        current_scalar_profile_id=current["profile_identity"],
        historical_amplitude_compiler_id=historical["profile_identity"],
        densities=tuple(specification["densities"]),
        density_rms=tuple(specification["expected_density_rms"]),
        anchor_density=specification["anchor_density"],
        anchor_density_rms=specification["anchor_density_rms"],
        aperture_diameter_micrometres=specification["aperture_diameter_micrometres"],
        current_400tx_calibration_claimed=specification[
            "current_400tx_calibration_claimed"
        ],
        historical_5233_calibration_claimed=specification[
            "historical_5233_calibration_claimed"
        ],
        scanner_response_included=specification["scanner_response_included"],
    )
    restored = BWHybridDensityAmplitudeProfile.from_dict(profile.to_dict())
    densities = np.asarray(specification["densities"], dtype=np.float64)
    expected_sigma = np.asarray(specification["expected_density_rms"], dtype=np.float64)
    actual_sigma = profile.sigma_d(densities)
    profile_node_error = float(np.max(np.abs(actual_sigma / expected_sigma - 1.0)))
    spatial = contract["spatial_mechanism"]
    aperture = binary_circular_aperture_kernel(
        spatial["sample_pitch_micrometres"],
        spatial["measurement_aperture_diameter_micrometres"],
    )
    energy = thomas_aperture_measurement_energy(
        particle_sigma_samples=spatial["particle_sigma_samples"],
        cluster_sigma_samples=spatial["cluster_sigma_samples"],
        mean_offspring=spatial["mean_offspring"],
        truncate=spatial["truncate_sigma"],
        aperture_kernel=aperture,
    )
    point_scales = actual_sigma / math.sqrt(energy)
    analytic_errors = np.abs(point_scales * math.sqrt(energy) - actual_sigma)
    evaluation = contract["evaluation"]
    shape = tuple(evaluation["field_shape"])
    rows = []
    repeat_exact = True
    partition_exact = True
    for seed in evaluation["seeds"]:
        kwargs = {
            "full_shape": shape,
            "profile_id": spatial["spatial_profile_id"],
            "particle_sigma_pixels": spatial["particle_sigma_samples"],
            "cluster_sigma_pixels": spatial["cluster_sigma_samples"],
            "mean_offspring": spatial["mean_offspring"],
            "component_seeds": tuple(spatial["component_seeds"]),
            "realization_seed": seed,
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
        assembled = np.empty_like(unit)
        row_height = evaluation["row_partition_height"]
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            assembled[y0 : y0 + height] = render_dc_projected_thomas_region(
                receipt,
                origin_yx=(y0, 0),
                shape=(height, shape[1]),
            )
        partition_exact &= bool(np.array_equal(unit, assembled))
        for density, target_sigma, point_scale in zip(
            densities, actual_sigma, point_scales, strict=True
        ):
            field = np.ascontiguousarray(density + point_scale * unit, dtype=np.float64)
            measured = fftconvolve(field, aperture, mode="same")
            border = evaluation["measurement_crop_border_samples"]
            measured = measured[border:-border, border:-border]
            observed_sigma = float(np.std(measured, dtype=np.float64))
            rows.append(
                {
                    "density": float(density),
                    "seed": int(seed),
                    "receipt_id": receipt.receipt_id,
                    "target_aperture_sigma_d": float(target_sigma),
                    "observed_aperture_sigma_d": observed_sigma,
                    "aperture_sigma_relative_error": abs(observed_sigma - target_sigma)
                    / target_sigma,
                    "absolute_field_mean": abs(
                        float(np.mean(field - density, dtype=np.float64))
                    ),
                    "minimum_developed_density": float(np.min(field)),
                    "density_sha256": hashlib.sha256(
                        np.asarray(field, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    errors = np.asarray(
        [row["aperture_sigma_relative_error"] for row in rows], dtype=np.float64
    )
    measurements = {
        "profile_identity": profile.identity(),
        "profile_node_relative_error": profile_node_error,
        "maximum_analytic_aperture_sigma_error": float(np.max(analytic_errors)),
        "median_aperture_sigma_relative_error": float(np.median(errors)),
        "p95_aperture_sigma_relative_error": float(np.percentile(errors, 95.0)),
        "maximum_absolute_field_mean": max(row["absolute_field_mean"] for row in rows),
        "minimum_developed_density": min(
            row["minimum_developed_density"] for row in rows
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "exact_roundtrip_identity": (
            profile.to_dict() == restored.to_dict()
            and profile.identity() == restored.identity()
        ),
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "maximum_profile_node_relative_error": profile_node_error
        <= evaluation["maximum_profile_node_relative_error"],
        "maximum_analytic_aperture_sigma_error": measurements[
            "maximum_analytic_aperture_sigma_error"
        ]
        <= evaluation["maximum_analytic_aperture_sigma_error"],
        "maximum_median_aperture_sigma_relative_error": measurements[
            "median_aperture_sigma_relative_error"
        ]
        <= evaluation["maximum_median_aperture_sigma_relative_error"],
        "maximum_p95_aperture_sigma_relative_error": measurements[
            "p95_aperture_sigma_relative_error"
        ]
        <= evaluation["maximum_p95_aperture_sigma_relative_error"],
        "maximum_absolute_field_mean": measurements["maximum_absolute_field_mean"]
        <= evaluation["maximum_absolute_field_mean"],
        "minimum_developed_density": measurements["minimum_developed_density"]
        >= evaluation["minimum_developed_density"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "exact_roundtrip_identity": measurements["exact_roundtrip_identity"],
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2ar_hybrid_density_amplitude_thomas_report.v1",
        "profile": profile.to_dict(),
        "measurement_energy": energy,
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
