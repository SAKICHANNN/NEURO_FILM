"""Evaluate cross-layer covariance after Gaussian cloud footprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_cross_layer_poisson_region,
)
from src.film_physics.cross_layer_gaussian_cloud import (
    discrete_gaussian_kernel_overlap,
    render_cross_layer_gaussian_cloud_density,
)
from src.film_physics.density_conditioned_structure import counter_poisson_rate_field

SCHEMA = "neuro_film.u6_p4da_cross_layer_gaussian_cloud_lod_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4da_cross_layer_gaussian_cloud_lod_report.v1"


class CrossLayerGaussianCloudError(RuntimeError):
    """Raised when the frozen P4DA experiment drifts."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload["parent"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise CrossLayerGaussianCloudError("P4DA contract identity drift")
    if (
        sha256_file(parent_path) != parent["sha256"]
        or parent_payload.get("decision") != parent["required_decision"]
    ):
        raise CrossLayerGaussianCloudError("P4DA parent drift")
    field = payload["field"]
    if field.get("shape") != [512, 768] or field.get("boundary_mode") != "wrap":
        raise CrossLayerGaussianCloudError("P4DA field drift")
    return payload


def _profile(field: dict[str, Any], seed: int) -> CrossLayerPoissonProfile:
    return CrossLayerPoissonProfile(
        tuple(field["marginal_count_rates_cmy"]),
        field["shared_all_rate"],
        tuple(field["shared_pair_rates_cm_cy_my"]),
        tuple(field["mark_optical_density_cmy"]),
        seed,
        field["component_seed_stride"],
    )


def _analytic(
    profile: CrossLayerPoissonProfile,
    sigmas: tuple[float, float, float],
    truncate: float,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(profile.marginal_rates_cmy, dtype=np.float64)
    all_rate = profile.shared_all_rate
    cm, cy, my = profile.shared_pair_rates_cm_cy_my
    shared = np.asarray(
        [
            [rates[0], all_rate + cm, all_rate + cy],
            [all_rate + cm, rates[1], all_rate + my],
            [all_rate + cy, all_rate + my, rates[2]],
        ]
    )
    overlap = np.asarray(
        [
            [
                discrete_gaussian_kernel_overlap(
                    sigmas[i], sigmas[j], truncate=truncate
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    covariance = shared * overlap
    variance = np.diag(covariance)
    correlation = covariance / np.sqrt(variance[:, None] * variance[None, :])
    np.fill_diagonal(correlation, 1.0)
    return variance, correlation


def _independent_counts(
    profile: CrossLayerPoissonProfile, shape: tuple[int, int]
) -> np.ndarray:
    layers = []
    for index, rate in enumerate(profile.marginal_rates_cmy):
        layers.append(
            counter_poisson_rate_field(
                np.full(shape, rate),
                shape,
                origin_yx=(0, 0),
                seed=profile.seed + (index + 20) * profile.component_seed_stride,
                maximum_rate=rate,
            )
        )
    return np.stack(layers, axis=-1)


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, 3).astype(np.float64)
    return np.var(flat, axis=0), np.corrcoef(flat, rowvar=False)


def _row(seed: int, contract: dict[str, Any]) -> dict[str, Any]:
    field = contract["field"]
    shape = tuple(field["shape"])
    profile = _profile(field, seed)
    sigmas = tuple(field["gaussian_sigma_pixels_cmy"])
    truncate = field["gaussian_truncate"]
    counts = sample_cross_layer_poisson_region(
        profile, shape, origin_yx=(0, 0), shape=shape
    )
    density = render_cross_layer_gaussian_cloud_density(
        counts,
        mark_optical_density_cmy=profile.mark_optical_density_cmy,
        sigma_pixels_cmy=sigmas,
        truncate=truncate,
    )
    repeat = render_cross_layer_gaussian_cloud_density(
        counts,
        mark_optical_density_cmy=profile.mark_optical_density_cmy,
        sigma_pixels_cmy=sigmas,
        truncate=truncate,
    )
    variance, correlation = _stats(density)
    analytic_variance, analytic_correlation = _analytic(profile, sigmas, truncate)
    independent = render_cross_layer_gaussian_cloud_density(
        _independent_counts(profile, shape),
        mark_optical_density_cmy=profile.mark_optical_density_cmy,
        sigma_pixels_cmy=sigmas,
        truncate=truncate,
    )
    _, independent_correlation = _stats(independent)
    offdiag = np.triu_indices(3, 1)
    return {
        "seed": seed,
        "maximum_absolute_correlation_error": float(
            np.max(np.abs(correlation[offdiag] - analytic_correlation[offdiag]))
        ),
        "maximum_marginal_variance_relative_error": float(
            np.max(
                np.abs(
                    variance
                    - analytic_variance * np.square(profile.mark_optical_density_cmy)
                )
                / (analytic_variance * np.square(profile.mark_optical_density_cmy))
            )
        ),
        "minimum_empirical_shared_correlation": float(np.min(correlation[offdiag])),
        "maximum_independent_control_absolute_correlation": float(
            np.max(np.abs(independent_correlation[offdiag]))
        ),
        "repeat_exact": bool(np.array_equal(density, repeat)),
        "density_minimum": float(np.min(density)),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    c = load_contract(root, contract_path)
    field = c["field"]
    development = [_row(int(s), c) for s in field["development_seeds"]]
    confirmation = [_row(int(s), c) for s in field["confirmation_seeds"]]
    m = c["metrics"]
    worst = {
        "maximum_absolute_correlation_error": max(
            r["maximum_absolute_correlation_error"] for r in confirmation
        ),
        "maximum_marginal_variance_relative_error": max(
            r["maximum_marginal_variance_relative_error"] for r in confirmation
        ),
        "maximum_independent_control_absolute_correlation": max(
            r["maximum_independent_control_absolute_correlation"] for r in confirmation
        ),
        "minimum_empirical_shared_correlation": min(
            r["minimum_empirical_shared_correlation"] for r in confirmation
        ),
    }
    gates = {
        "correlation": worst["maximum_absolute_correlation_error"]
        <= m["maximum_absolute_correlation_error"],
        "variance": worst["maximum_marginal_variance_relative_error"]
        <= m["maximum_marginal_variance_relative_error"],
        "independent": worst["maximum_independent_control_absolute_correlation"]
        <= m["maximum_independent_control_absolute_correlation"],
        "shared_signal": worst["minimum_empirical_shared_correlation"]
        >= m["minimum_shared_correlation"],
        "repeat": all(r["repeat_exact"] for r in (*development, *confirmation)),
        "nonnegative": all(
            r["density_minimum"] >= 0 for r in (*development, *confirmation)
        ),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "confirmation_worst_metrics": worst,
        "gates": gates,
        "decision": c["decision_if_pass"] if passed else c["decision_if_fail"],
        "claim_ceiling": c["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "development": development,
        "confirmation": confirmation,
    }


__all__ = ["CrossLayerGaussianCloudError", "evaluate", "load_contract"]
