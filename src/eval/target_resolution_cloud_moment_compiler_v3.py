"""Frozen U6.P4DT moment compiler for target-resolution cloud LOD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cloud_spatial_order_ablation import _edge_excursion
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.target_resolution_cloud_lod_v2 import (
    _noise_stats,
    _radial_nps,
    _render,
    compile_target_resolution_profile,
)
from src.film_physics.cross_layer_aperture_lod import block_aperture_mean
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
        or evidence["failed_gate"] != contract["parent"]["required_failed_gate"]
    ):
        raise RuntimeError("P4DT parent drift")
    return contract


def _pair_metrics(
    reference: np.ndarray, candidate: np.ndarray, expected: np.ndarray
) -> dict[str, float]:
    ref_noise = reference - expected
    candidate_noise = candidate - expected
    ref_rms, ref_corr = _noise_stats(ref_noise)
    candidate_rms, candidate_corr = _noise_stats(candidate_noise)
    index = np.triu_indices(3, 1)
    return {
        "mean_transmittance_absolute_error": float(
            np.max(
                np.abs(
                    np.mean(candidate, axis=(0, 1)) - np.mean(reference, axis=(0, 1))
                )
            )
        ),
        "marginal_noise_rms_relative_error": float(
            np.max(np.abs(candidate_rms - ref_rms) / ref_rms)
        ),
        "cross_channel_correlation_absolute_error": float(
            np.max(np.abs(candidate_corr[index] - ref_corr[index]))
        ),
        "radial_nps_shape_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(_radial_nps(candidate_noise) - _radial_nps(ref_noise))
                )
            )
        ),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    factor = fixture["aperture_factor"]
    shape = (fixture["output_height"], fixture["output_width"])
    high_shape = (shape[0] * factor, shape[1] * factor)
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    candidates = {
        multiplier: compile_target_resolution_profile(
            reference, factor, rate_multiplier=multiplier
        )
        for multiplier in contract["compiler"]["candidate_rate_multipliers"]
    }
    common_capacity = np.min(
        np.stack(
            [
                np.asarray(optical_density_capacity_cmy(profile))
                for profile in (reference, *candidates.values())
            ]
        ),
        axis=0,
    )
    target_value = common_capacity * fixture["flat_density_fraction"]
    target = np.broadcast_to(target_value, (*shape, 3)).copy()
    high_target = np.broadcast_to(target_value, (*high_shape, 3)).copy()
    expected = np.power(10.0, -target_value)
    development_rows = []
    for seed in fixture["development_seeds"]:
        ref = block_aperture_mean(_render(reference, high_target, seed), factor)
        for multiplier, candidate in candidates.items():
            metrics = _pair_metrics(ref, _render(candidate, target, seed), expected)
            development_rows.append(
                {"seed": seed, "rate_multiplier": multiplier, **metrics}
            )
    summaries = []
    for multiplier in candidates:
        rows = [row for row in development_rows if row["rate_multiplier"] == multiplier]
        summaries.append(
            {
                "rate_multiplier": multiplier,
                "development_median_noise_rms_relative_error": float(
                    np.median(
                        [row["marginal_noise_rms_relative_error"] for row in rows]
                    )
                ),
            }
        )
    selected = min(
        summaries,
        key=lambda row: (
            row["development_median_noise_rms_relative_error"],
            row["rate_multiplier"],
        ),
    )
    multiplier = selected["rate_multiplier"]
    candidate = candidates[multiplier]
    confirmation = []
    for seed in fixture["confirmation_seeds"]:
        ref = block_aperture_mean(_render(reference, high_target, seed), factor)
        observed = _render(candidate, target, seed)
        confirmation.append({"seed": seed, **_pair_metrics(ref, observed, expected)})
    edge_target = np.empty((*shape, 3), dtype=np.float64)
    edge_target[:, : shape[1] // 2] = common_capacity * 0.2
    edge_target[:, shape[1] // 2 :] = common_capacity * 0.8
    high_edge_target = np.repeat(np.repeat(edge_target, factor, axis=0), factor, axis=1)
    ref_edge = block_aperture_mean(
        _render(reference, high_edge_target, fixture["development_seeds"][0]), factor
    )
    candidate_edge = _render(candidate, edge_target, fixture["development_seeds"][0])
    candidate_repeat = _render(candidate, edge_target, fixture["development_seeds"][0])
    ref_over, ref_under = _edge_excursion(ref_edge)
    cand_over, cand_under = _edge_excursion(candidate_edge)
    worst = {
        "development_noise_rms_relative_error": selected[
            "development_median_noise_rms_relative_error"
        ],
        "confirmation_mean_transmittance_absolute_error": max(
            row["mean_transmittance_absolute_error"] for row in confirmation
        ),
        "confirmation_marginal_noise_rms_relative_error": max(
            row["marginal_noise_rms_relative_error"] for row in confirmation
        ),
        "confirmation_cross_channel_correlation_absolute_error": max(
            row["cross_channel_correlation_absolute_error"] for row in confirmation
        ),
        "confirmation_radial_nps_shape_rmse": max(
            row["radial_nps_shape_rmse"] for row in confirmation
        ),
        "confirmation_edge_overshoot_absolute_difference": abs(cand_over - ref_over),
        "confirmation_edge_undershoot_absolute_difference": abs(cand_under - ref_under),
        "new_boundary_fraction": float(
            np.mean(
                ((candidate_edge <= 0.0) | (candidate_edge >= 1.0))
                & ~((ref_edge <= 0.0) | (ref_edge >= 1.0))
            )
        ),
    }
    gates = contract["gates"]
    decisions = {
        "development": worst["development_noise_rms_relative_error"]
        <= gates["maximum_development_noise_rms_relative_error"],
        "mean": worst["confirmation_mean_transmittance_absolute_error"]
        <= gates["maximum_confirmation_mean_transmittance_absolute_error"],
        "rms": worst["confirmation_marginal_noise_rms_relative_error"]
        <= gates["maximum_confirmation_marginal_noise_rms_relative_error"],
        "correlation": worst["confirmation_cross_channel_correlation_absolute_error"]
        <= gates["maximum_confirmation_cross_channel_correlation_absolute_error"],
        "nps": worst["confirmation_radial_nps_shape_rmse"]
        <= gates["maximum_confirmation_radial_nps_shape_rmse"],
        "overshoot": worst["confirmation_edge_overshoot_absolute_difference"]
        <= gates["maximum_confirmation_edge_overshoot_absolute_difference"],
        "undershoot": worst["confirmation_edge_undershoot_absolute_difference"]
        <= gates["maximum_confirmation_edge_undershoot_absolute_difference"],
        "boundary": worst["new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "repeat": bool(np.array_equal(candidate_edge, candidate_repeat)),
        "product_disabled": candidate.product_enabled is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "reference_profile_identity": reference.identity(),
        "selected_rate_multiplier": multiplier,
        "selected_profile_identity": candidate.identity(),
        "development_summaries": summaries,
        "confirmation_worst": worst,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dt_target_resolution_cloud_moment_compiler_report.v3",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "development": development_rows,
        "confirmation": confirmation,
    }


__all__ = ["evaluate", "load_contract"]
