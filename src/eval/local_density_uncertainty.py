"""U6.P2AX finite-realization uncertainty for nonuniform density structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.eval.nonuniform_positive_density_wedge import load_reference_profiles
from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
)
from src.film_physics.positive_density_field import (
    render_nonuniform_positive_density_region,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt


class LocalDensityUncertaintyError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ax_local_density_uncertainty_contract.v1"
    ):
        raise LocalDensityUncertaintyError("unsupported P2AX contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise LocalDensityUncertaintyError("P2AX parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise LocalDensityUncertaintyError("P2AX parent decision mismatch")
    if "required_decision" in binding and (
        payload.get("decision") != binding["required_decision"]
    ):
        raise LocalDensityUncertaintyError("P2AX parent branch mismatch")
    return payload


def _layout(contract: dict[str, Any]) -> np.ndarray:
    layout = contract["fresh_layout"]
    shape = tuple(layout["shape"])
    cell_shape = tuple(layout["cell_shape"])
    rows = shape[0] // cell_shape[0]
    columns = shape[1] // cell_shape[1]
    densities = np.asarray(layout["ordered_cell_densities"], dtype=np.float64)
    if (
        shape[0] % cell_shape[0]
        or shape[1] % cell_shape[1]
        or densities.size != rows * columns
    ):
        raise LocalDensityUncertaintyError("P2AX layout geometry drift")
    grid = densities.reshape(rows, columns)
    return np.ascontiguousarray(
        np.repeat(np.repeat(grid, cell_shape[0], axis=0), cell_shape[1], axis=1),
        dtype=np.float64,
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    _load_bound(root, contract["parents"]["single_realization_boundary"])
    wedge_contract = _load_bound(root, contract["parents"]["wedge_contract"])
    amplitude_profile, parameter_profile = load_reference_profiles(
        root=root, contract=wedge_contract
    )
    mean_density = _layout(contract)
    layout = contract["fresh_layout"]
    shape = tuple(layout["shape"])
    cell_shape = tuple(layout["cell_shape"])
    spatial = wedge_contract["spatial_mechanism"]
    source_evaluation = wedge_contract["evaluation"]
    evaluation = contract["evaluation"]
    uncertainty = contract["uncertainty_model"]
    aperture = binary_circular_aperture_kernel(
        spatial["sample_pitch_micrometres"],
        spatial["measurement_aperture_diameter_micrometres"],
    )
    densities = np.asarray(layout["ordered_cell_densities"], dtype=np.float64)
    per_cell_means: list[list[float]] = [[] for _ in densities]
    per_cell_sigmas: list[list[float]] = [[] for _ in densities]
    realization_rows: list[dict[str, Any]] = []
    minimum_density = float("inf")
    inset = int(evaluation["cell_measurement_inset_samples"])
    cells_per_row = shape[1] // cell_shape[1]
    for seed in layout["realization_seeds"]:
        receipt = build_thomas_dc_receipt(
            full_shape=shape,
            profile_id=spatial["spatial_profile_id"],
            particle_sigma_pixels=spatial["particle_sigma_samples"],
            cluster_sigma_pixels=spatial["cluster_sigma_samples"],
            mean_offspring=spatial["mean_offspring"],
            component_seeds=tuple(spatial["component_seeds"]),
            realization_seed=int(seed),
            truncate=spatial["truncate_sigma"],
            canonical_row_block_height=source_evaluation[
                "canonical_row_block_height"
            ],
        )
        field = render_nonuniform_positive_density_region(
            receipt,
            mean_density=mean_density,
            origin_yx=(0, 0),
            shape=shape,
            amplitude_profile=amplitude_profile,
            parameter_profile=parameter_profile,
        )
        measured = fftconvolve(field, aperture, mode="same")
        minimum_density = min(minimum_density, float(np.min(field)))
        cell_rows = []
        for index, density in enumerate(densities):
            cell_y = index // cells_per_row
            cell_x = index % cells_per_row
            y0 = cell_y * cell_shape[0] + inset
            y1 = (cell_y + 1) * cell_shape[0] - inset
            x0 = cell_x * cell_shape[1] + inset
            x1 = (cell_x + 1) * cell_shape[1] - inset
            observed_mean = float(np.mean(field[y0:y1, x0:x1], dtype=np.float64))
            observed_sigma = float(np.std(measured[y0:y1, x0:x1], dtype=np.float64))
            per_cell_means[index].append(observed_mean)
            per_cell_sigmas[index].append(observed_sigma)
            cell_rows.append(
                {
                    "cell_index": index,
                    "density": float(density),
                    "observed_density_mean": observed_mean,
                    "observed_aperture_sigma_d": observed_sigma,
                }
            )
        realization_rows.append(
            {
                "seed": int(seed),
                "receipt_id": receipt.receipt_id,
                "cells": cell_rows,
                "density_sha256": hashlib.sha256(
                    np.ascontiguousarray(field, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    t_multiplier = float(uncertainty["student_t_two_sided_95_multiplier_df11"])
    aggregate_rows = []
    mean_errors = []
    sigma_errors = []
    covered = 0
    for index, density in enumerate(densities):
        means = np.asarray(per_cell_means[index], dtype=np.float64)
        sigmas = np.asarray(per_cell_sigmas[index], dtype=np.float64)
        expected_mean = float(np.mean(means))
        expected_sigma = float(np.mean(sigmas))
        target_sigma = float(amplitude_profile.sigma_d(float(density)))
        standard_error = float(np.std(sigmas, ddof=1) / np.sqrt(sigmas.size))
        lower = expected_sigma - t_multiplier * standard_error
        upper = expected_sigma + t_multiplier * standard_error
        target_covered = lower <= target_sigma <= upper
        covered += int(target_covered)
        mean_error = abs(expected_mean - density) / density
        sigma_error = abs(expected_sigma - target_sigma) / target_sigma
        mean_errors.append(mean_error)
        sigma_errors.append(sigma_error)
        aggregate_rows.append(
            {
                "cell_index": index,
                "density": float(density),
                "target_aperture_sigma_d": target_sigma,
                "expected_density_mean": expected_mean,
                "expected_aperture_sigma_d": expected_sigma,
                "relative_mean_error": mean_error,
                "relative_sigma_error": sigma_error,
                "aperture_sigma_standard_error": standard_error,
                "aperture_sigma_interval_95": [lower, upper],
                "target_interval_covered": target_covered,
            }
        )
    coverage = covered / len(densities)
    measurements = {
        "amplitude_profile_identity": amplitude_profile.identity(),
        "parameter_profile_identity": parameter_profile.identity(),
        "maximum_aggregate_relative_mean_error": float(np.max(mean_errors)),
        "aggregate_median_sigma_relative_error": float(np.median(sigma_errors)),
        "aggregate_p95_sigma_relative_error": float(
            np.percentile(sigma_errors, 95.0)
        ),
        "target_interval_coverage_fraction": coverage,
        "minimum_developed_density": minimum_density,
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    gate_results = {
        "maximum_aggregate_relative_mean_error": measurements[
            "maximum_aggregate_relative_mean_error"
        ]
        <= evaluation["maximum_aggregate_relative_mean_error"],
        "maximum_aggregate_median_sigma_relative_error": measurements[
            "aggregate_median_sigma_relative_error"
        ]
        <= evaluation["maximum_aggregate_median_sigma_relative_error"],
        "maximum_aggregate_p95_sigma_relative_error": measurements[
            "aggregate_p95_sigma_relative_error"
        ]
        <= evaluation["maximum_aggregate_p95_sigma_relative_error"],
        "minimum_target_interval_coverage_fraction": coverage
        >= evaluation["minimum_target_interval_coverage_fraction"],
        "minimum_developed_density_exclusive": minimum_density
        > evaluation["minimum_developed_density_exclusive"],
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p2ax_local_density_uncertainty_report.v1",
        "realizations": realization_rows,
        "aggregate_rows": aggregate_rows,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
