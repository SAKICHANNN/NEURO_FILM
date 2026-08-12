"""Frozen U6.P4DC density-conditioned shared-count evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    density_conditioned_component_rates,
    sample_density_conditioned_cross_layer_poisson_region,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
        or evidence["profile_identity"] != contract["parent"]["profile_identity"]
    ):
        raise RuntimeError("P8DF parent drift")
    return contract


def _profile(root: Path, seed: int) -> CrossLayerPoissonProfile:
    p = json.loads(
        (
            root / "configs/u6_p8df_cross_layer_cloud_profile_compiler_v1.json"
        ).read_text()
    )["profile"]
    return CrossLayerPoissonProfile(
        tuple(p["marginal_count_rates_cmy"]),
        p["shared_all_rate"],
        tuple(p["shared_pair_rates_cm_cy_my"]),
        tuple(p["mark_optical_density_cmy"]),
        seed,
        p["component_seed_stride"],
    )


def _expected(
    scale: np.ndarray, profile: CrossLayerPoissonProfile
) -> tuple[np.ndarray, np.ndarray]:
    component = density_conditioned_component_rates(profile, scale[None, None, :])[0, 0]
    means = np.asarray(profile.marginal_rates_cmy) * scale
    covariance = np.diag(means)
    covariance[0, 1] = covariance[1, 0] = component[0] + component[1]
    covariance[0, 2] = covariance[2, 0] = component[0] + component[2]
    covariance[1, 2] = covariance[2, 1] = component[0] + component[3]
    return means, covariance


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    qh, qw = fixture["quadrant_height"], fixture["quadrant_width"]
    shape = (2 * qh, 2 * qw)
    scale = np.empty((*shape, 3), dtype=np.float64)
    quadrants = (
        (slice(0, qh), slice(0, qw)),
        (slice(0, qh), slice(qw, 2 * qw)),
        (slice(qh, 2 * qh), slice(0, qw)),
        (slice(qh, 2 * qh), slice(qw, 2 * qw)),
    )
    for region, values in zip(quadrants, fixture["scale_cmy_by_quadrant"], strict=True):
        scale[region] = values
    profile = _profile(root, fixture["seed"])
    full = sample_density_conditioned_cross_layer_poisson_region(
        profile, scale, shape, origin_yx=(0, 0)
    )
    pieces = []
    tile = fixture["row_partition_height"]
    for y0 in range(0, shape[0], tile):
        y1 = min(shape[0], y0 + tile)
        pieces.append(
            sample_density_conditioned_cross_layer_poisson_region(
                profile, scale[y0:y1], shape, origin_yx=(y0, 0)
            )
        )
    partitioned = np.concatenate(pieces, axis=0)
    metrics = []
    for index, (region, values) in enumerate(
        zip(quadrants, fixture["scale_cmy_by_quadrant"], strict=True)
    ):
        observed = full[region].reshape(-1, 3).astype(np.float64)
        observed_mean = np.mean(observed, axis=0)
        observed_covariance = np.cov(observed, rowvar=False, ddof=0)
        expected_mean, expected_covariance = _expected(np.asarray(values), profile)
        expected_corr = expected_covariance / np.sqrt(
            np.outer(expected_mean, expected_mean)
        )
        observed_corr = observed_covariance / np.sqrt(
            np.outer(np.diag(observed_covariance), np.diag(observed_covariance))
        )
        offdiag = ~np.eye(3, dtype=bool)
        metrics.append(
            {
                "quadrant": index,
                "maximum_mean_relative_error": float(
                    np.max(np.abs(observed_mean - expected_mean) / expected_mean)
                ),
                "maximum_covariance_relative_error": float(
                    np.max(
                        np.abs(
                            observed_covariance[offdiag] - expected_covariance[offdiag]
                        )
                        / expected_covariance[offdiag]
                    )
                ),
                "maximum_correlation_absolute_error": float(
                    np.max(np.abs(observed_corr[offdiag] - expected_corr[offdiag]))
                ),
                "minimum_component_rate": float(
                    np.min(
                        density_conditioned_component_rates(
                            profile, np.asarray(values)[None, None, :]
                        )
                    )
                ),
            }
        )
    gates = contract["gates"]
    gate_results = {
        "marginal_mean": max(x["maximum_mean_relative_error"] for x in metrics)
        <= gates["maximum_marginal_mean_relative_error"],
        "covariance": max(x["maximum_covariance_relative_error"] for x in metrics)
        <= gates["maximum_covariance_relative_error"],
        "correlation": max(x["maximum_correlation_absolute_error"] for x in metrics)
        <= gates["maximum_correlation_absolute_error"],
        "nonnegative_rates": min(x["minimum_component_rate"] for x in metrics) >= 0.0,
        "partition_identity": bool(np.array_equal(full, partitioned)),
        "repeat_identity": bool(
            np.array_equal(
                full,
                sample_density_conditioned_cross_layer_poisson_region(
                    profile, scale, shape, origin_yx=(0, 0)
                ),
            )
        ),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": contract["parent"]["profile_identity"],
        "count_sha256": hashlib.sha256(full.astype("<u2").tobytes()).hexdigest(),
        "quadrants": metrics,
        "gates": gate_results,
        "decision": contract["decision_if_pass"]
        if all(gate_results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dc_density_conditioned_cross_layer_counts_report.v1",
        "automatic_pass": all(gate_results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
