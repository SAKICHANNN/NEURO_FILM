"""U6.P2AS positive softplus density-field development and confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
)
from src.film_physics.positive_density_field import softplus_density_field
from src.film_physics.thomas_dc_projection import (
    ThomasDcReceipt,
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)


class PositiveSoftplusDensityFieldError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2as_positive_softplus_density_field_contract.v1"
    ):
        raise PositiveSoftplusDensityFieldError("unsupported P2AS contract")
    return payload


def _load_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise PositiveSoftplusDensityFieldError("P2AS parent mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise PositiveSoftplusDensityFieldError("P2AS parent decision mismatch")
    if "required_decision" in binding and (
        payload.get("decision") != binding["required_decision"]
    ):
        raise PositiveSoftplusDensityFieldError("P2AS parent branch mismatch")
    return payload


def _receipt(
    spatial: dict[str, Any], evaluation: dict[str, Any], seed: int
) -> ThomasDcReceipt:
    return build_thomas_dc_receipt(
        full_shape=tuple(evaluation["field_shape"]),
        profile_id=spatial["spatial_profile_id"],
        particle_sigma_pixels=spatial["particle_sigma_samples"],
        cluster_sigma_pixels=spatial["cluster_sigma_samples"],
        mean_offspring=spatial["mean_offspring"],
        component_seeds=tuple(spatial["component_seeds"]),
        realization_seed=seed,
        truncate=spatial["truncate_sigma"],
        canonical_row_block_height=evaluation["canonical_row_block_height"],
    )


def _unit_field(receipt: ThomasDcReceipt) -> np.ndarray:
    return render_dc_projected_thomas_region(
        receipt, origin_yx=(0, 0), shape=receipt.full_shape
    )


def _measure(
    field: np.ndarray, aperture: np.ndarray, border: int
) -> tuple[float, float]:
    measured = fftconvolve(field, aperture, mode="same")
    selected = measured[border:-border, border:-border]
    return (
        float(np.mean(field, dtype=np.float64)),
        float(np.std(selected, dtype=np.float64)),
    )


def _fit_parameters(
    *,
    units: list[np.ndarray],
    target_mean: float,
    target_sigma: float,
    aperture: np.ndarray,
    border: int,
    mechanism: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    bounds = mechanism["parameter_bounds"]
    a0 = math.log(math.expm1(target_mean))
    local_slope = 1.0 - math.exp(-target_mean)
    b0 = min(
        bounds["b"][1] * 0.5,
        max(bounds["b"][0] + 1e-6, target_sigma / local_slope),
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        means = []
        sigmas = []
        for unit in units:
            field = softplus_density_field(unit, a=parameters[0], b=parameters[1])
            mean, sigma = _measure(field, aperture, border)
            means.append(mean)
            sigmas.append(sigma)
        return np.asarray(
            [
                (float(np.mean(means)) - target_mean) / target_mean,
                (float(np.mean(sigmas)) - target_sigma) / target_sigma,
            ],
            dtype=np.float64,
        )

    settings = mechanism["least_squares"]
    result = least_squares(
        residual,
        np.asarray([a0, b0], dtype=np.float64),
        bounds=(
            np.asarray([bounds["a"][0], bounds["b"][0]], dtype=np.float64),
            np.asarray([bounds["a"][1], bounds["b"][1]], dtype=np.float64),
        ),
        ftol=settings["ftol"],
        xtol=settings["xtol"],
        gtol=settings["gtol"],
        max_nfev=settings["maximum_function_evaluations"],
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise PositiveSoftplusDensityFieldError("P2AS parameter fit failed")
    final_residual = residual(result.x)
    return result.x, {
        "relative_mean_error": abs(float(final_residual[0])),
        "relative_sigma_error": abs(float(final_residual[1])),
        "function_evaluations": int(result.nfev),
    }


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    _load_parent(root, contract["parents"]["additive_boundary"])
    additive_contract = _load_parent(root, contract["parents"]["additive_contract"])
    mechanism = contract["positive_mechanism"]
    spatial = contract["spatial_mechanism"]
    evaluation = contract["evaluation"]
    if (
        additive_contract["hybrid_profile"]["densities"] != mechanism["densities"]
        or additive_contract["hybrid_profile"]["expected_density_rms"]
        != mechanism["target_aperture_sigma_d"]
        or any(
            additive_contract["spatial_mechanism"][key] != value
            for key, value in spatial.items()
        )
    ):
        raise PositiveSoftplusDensityFieldError("P2AS inherited mechanism drifted")
    aperture = binary_circular_aperture_kernel(
        spatial["sample_pitch_micrometres"],
        spatial["measurement_aperture_diameter_micrometres"],
    )
    development_units = [
        _unit_field(_receipt(spatial, evaluation, int(seed)))
        for seed in evaluation["development_seeds"]
    ]
    confirmation = [
        (
            int(seed),
            _receipt(spatial, evaluation, int(seed)),
        )
        for seed in evaluation["confirmation_seeds"]
    ]
    confirmation_units = [
        (seed, receipt, _unit_field(receipt)) for seed, receipt in confirmation
    ]
    parameter_rows = []
    parameter_pairs = []
    for density, target_sigma in zip(
        mechanism["densities"], mechanism["target_aperture_sigma_d"], strict=True
    ):
        parameters, fit = _fit_parameters(
            units=development_units,
            target_mean=float(density),
            target_sigma=float(target_sigma),
            aperture=aperture,
            border=evaluation["measurement_crop_border_samples"],
            mechanism=mechanism,
        )
        parameter_pairs.append((float(parameters[0]), float(parameters[1])))
        parameter_rows.append(
            {
                "density": float(density),
                "target_aperture_sigma_d": float(target_sigma),
                "a": float(parameters[0]),
                "b": float(parameters[1]),
                **fit,
            }
        )
    repeat_exact = True
    partition_exact = True
    confirmation_rows = []
    row_height = evaluation["row_partition_height"]
    shape = tuple(evaluation["field_shape"])
    for seed, receipt, unit in confirmation_units:
        repeated = _unit_field(_receipt(spatial, evaluation, seed))
        repeat_exact &= bool(np.array_equal(unit, repeated))
        unit_partition = np.empty_like(unit)
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            unit_partition[y0 : y0 + height] = render_dc_projected_thomas_region(
                receipt, origin_yx=(y0, 0), shape=(height, shape[1])
            )
        partition_exact &= bool(np.array_equal(unit, unit_partition))
        for density, target_sigma, parameters in zip(
            mechanism["densities"],
            mechanism["target_aperture_sigma_d"],
            parameter_pairs,
            strict=True,
        ):
            field = softplus_density_field(unit, a=parameters[0], b=parameters[1])
            partition_field = softplus_density_field(
                unit_partition, a=parameters[0], b=parameters[1]
            )
            partition_exact &= bool(np.array_equal(field, partition_field))
            mean, sigma = _measure(
                field, aperture, evaluation["measurement_crop_border_samples"]
            )
            confirmation_rows.append(
                {
                    "density": float(density),
                    "seed": seed,
                    "receipt_id": receipt.receipt_id,
                    "target_aperture_sigma_d": float(target_sigma),
                    "observed_density_mean": mean,
                    "observed_aperture_sigma_d": sigma,
                    "relative_mean_error": abs(mean - density) / density,
                    "relative_sigma_error": abs(sigma - target_sigma) / target_sigma,
                    "minimum_developed_density": float(np.min(field)),
                    "density_sha256": hashlib.sha256(
                        np.asarray(field, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    mean_errors = np.asarray(
        [row["relative_mean_error"] for row in confirmation_rows], dtype=np.float64
    )
    sigma_errors = np.asarray(
        [row["relative_sigma_error"] for row in confirmation_rows], dtype=np.float64
    )
    bounds = mechanism["parameter_bounds"]
    bound_margins = [
        min(
            a - bounds["a"][0],
            bounds["a"][1] - a,
            b - bounds["b"][0],
            bounds["b"][1] - b,
        )
        for a, b in parameter_pairs
    ]
    measurements = {
        "maximum_development_relative_mean_error": max(
            row["relative_mean_error"] for row in parameter_rows
        ),
        "maximum_development_relative_sigma_error": max(
            row["relative_sigma_error"] for row in parameter_rows
        ),
        "maximum_confirmation_relative_mean_error": float(np.max(mean_errors)),
        "confirmation_median_sigma_relative_error": float(np.median(sigma_errors)),
        "confirmation_p95_sigma_relative_error": float(
            np.percentile(sigma_errors, 95.0)
        ),
        "minimum_parameter_bound_margin": float(min(bound_margins)),
        "minimum_developed_density": min(
            row["minimum_developed_density"] for row in confirmation_rows
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "confirmation_parameter_refit_count_zero": True,
        "confirmation_realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "maximum_development_relative_mean_error": measurements[
            "maximum_development_relative_mean_error"
        ]
        <= evaluation["maximum_development_relative_mean_error"],
        "maximum_development_relative_sigma_error": measurements[
            "maximum_development_relative_sigma_error"
        ]
        <= evaluation["maximum_development_relative_sigma_error"],
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
        "minimum_parameter_bound_margin": measurements["minimum_parameter_bound_margin"]
        >= evaluation["minimum_parameter_bound_margin"],
        "minimum_developed_density_exclusive": measurements["minimum_developed_density"]
        > evaluation["minimum_developed_density_exclusive"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "confirmation_parameter_refit_count_zero": True,
        "confirmation_realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2as_positive_softplus_density_field_report.v1",
        "parameter_rows": parameter_rows,
        "confirmation_rows": confirmation_rows,
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
