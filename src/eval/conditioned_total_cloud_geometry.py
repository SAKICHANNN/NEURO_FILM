"""Evaluate conditioned-total binomial dye-cloud geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    build_conditioned_total_cloud_geometry,
)

SCHEMA = "neuro_film.u6_p4cz_conditioned_total_cloud_geometry_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cz_conditioned_total_cloud_geometry_report.v1"


class ConditionedTotalCloudError(RuntimeError):
    """Raised when the frozen P4CZ experiment drifts."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload["parent"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise ConditionedTotalCloudError("P4CZ contract identity drift")
    if (
        sha256_file(parent_path) != parent["sha256"]
        or parent_payload.get("decision") != parent["required_decision"]
    ):
        raise ConditionedTotalCloudError("P4CZ parent drift")
    geometry = payload["geometry"]
    if (
        geometry.get("input_shape") != [96, 128]
        or geometry.get("occupancy_cell_size") != 8
    ):
        raise ConditionedTotalCloudError("P4CZ geometry drift")
    return payload


def _profile(geometry: dict[str, Any], seed: int) -> CrossLayerPoissonProfile:
    return CrossLayerPoissonProfile(
        tuple(geometry["marginal_count_rates_cmy"]),
        geometry["shared_all_rate"],
        tuple(geometry["shared_pair_rates_cm_cy_my"]),
        tuple(geometry["mark_optical_density_cmy"]),
        seed,
        geometry["component_seed_stride"],
    )


def _occupancy(centers: np.ndarray, shape: tuple[int, int], cell: int) -> np.ndarray:
    rows, columns = shape[0] // cell, shape[1] // cell
    indexes_y = np.minimum((centers[:, 0] / cell).astype(np.int64), rows - 1)
    indexes_x = np.minimum((centers[:, 1] / cell).astype(np.int64), columns - 1)
    counts = np.bincount(indexes_y * columns + indexes_x, minlength=rows * columns)
    return counts.reshape(rows, columns)


def _row(seed: int, contract: dict[str, Any]) -> dict[str, Any]:
    geometry = contract["geometry"]
    shape = tuple(geometry["input_shape"])
    profile = _profile(geometry, seed)
    kwargs = {
        "radius_um_cmy": (4.0, 4.5, 5.0),
        "output_zoom": 1,
        "output_pixel_pitch_um": 4.0,
        "monte_carlo_samples": 1,
    }
    result = build_conditioned_total_cloud_geometry(profile, shape, **kwargs)
    repeat = build_conditioned_total_cloud_geometry(profile, shape, **kwargs)
    all_rate = profile.shared_all_rate
    cm, cy, my = profile.shared_pair_rates_cm_cy_my
    rates = np.asarray(
        (all_rate, cm, cy, my, *profile.independent_rates_cmy), dtype=np.float64
    )
    area = float(shape[0] * shape[1])
    observed_rates = np.asarray(
        [len(values) / area for values in result.component_centers]
    )
    cell = int(geometry["occupancy_cell_size"])
    cell_area = cell * cell
    cell_count = (shape[0] // cell) * (shape[1] // cell)
    mean_errors, variance_errors, correlations = [], [], []
    for rate, centers in zip(rates, result.component_centers, strict=True):
        occupancy = _occupancy(centers, shape, cell).astype(np.float64)
        expected_mean = rate * cell_area
        total = rate * area
        probability = 1.0 / cell_count
        expected_variance = total * probability * (1.0 - probability)
        mean_errors.append(
            abs(float(np.mean(occupancy)) - expected_mean) / expected_mean
        )
        variance_errors.append(
            abs(float(np.var(occupancy)) - expected_variance) / expected_variance
        )
        correlations.append(
            abs(
                float(
                    np.corrcoef(occupancy[:, :-1].ravel(), occupancy[:, 1:].ravel())[
                        0, 1
                    ]
                )
            )
        )
    shared_all_centers = result.component_centers[0]
    shared_identity = all(
        np.array_equal(layer[: len(shared_all_centers)], shared_all_centers)
        for layer in result.context.centers_by_layer
    )
    return {
        "seed": seed,
        "maximum_component_rate_relative_error": float(
            np.max(np.abs(observed_rates - rates) / rates)
        ),
        "maximum_cell_mean_relative_error": max(mean_errors),
        "maximum_cell_variance_relative_error": max(variance_errors),
        "maximum_absolute_nonoverlap_cell_correlation": max(correlations),
        "repeat_fingerprint_exact": result.context.fingerprint()
        == repeat.context.fingerprint(),
        "shared_center_identity_exact": shared_identity,
        "all_centers_in_bounds": all(
            np.all((values >= 0.0) & (values < np.asarray(shape)))
            for values in result.component_centers
        ),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    rows = [_row(int(seed), contract) for seed in contract["geometry"]["seeds"]]
    metrics = contract["metrics"]
    maxima = {
        key: max(float(row[key]) for row in rows)
        for key in (
            "maximum_component_rate_relative_error",
            "maximum_cell_mean_relative_error",
            "maximum_cell_variance_relative_error",
            "maximum_absolute_nonoverlap_cell_correlation",
        )
    }
    gates = {
        "rates": maxima["maximum_component_rate_relative_error"]
        <= metrics["maximum_component_rate_relative_error"],
        "cell_mean": maxima["maximum_cell_mean_relative_error"]
        <= metrics["maximum_cell_mean_relative_error"],
        "cell_variance": maxima["maximum_cell_variance_relative_error"]
        <= metrics["maximum_cell_variance_to_binomial_expectation_relative_error"],
        "cell_correlation": maxima["maximum_absolute_nonoverlap_cell_correlation"]
        <= metrics["maximum_absolute_nonoverlap_cell_correlation"],
        "repeat": all(row["repeat_fingerprint_exact"] for row in rows),
        "shared_identity": all(row["shared_center_identity_exact"] for row in rows),
        "bounds": all(row["all_centers_in_bounds"] for row in rows),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "maximum_metrics": maxima,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "rows": rows,
    }


__all__ = ["ConditionedTotalCloudError", "evaluate", "load_contract"]
