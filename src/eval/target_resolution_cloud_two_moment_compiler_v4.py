"""Frozen U6.P4DU two-moment target-resolution cloud compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

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
    ):
        raise RuntimeError("P4DU parent drift")
    return contract


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=(0, 1), dtype=np.float64))


def _comparison(
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
    compiler = contract["compiler"]
    factor = fixture["aperture_factor"]
    shape = (fixture["output_height"], fixture["output_width"])
    high_shape = (shape[0] * factor, shape[1] * factor)
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    bounds = tuple(compiler["rate_multiplier_bounds"])
    endpoint_profiles = [
        compile_target_resolution_profile(reference, factor, rate_multiplier=value)
        for value in bounds
    ]
    common_capacity = np.min(
        np.stack(
            [
                np.asarray(optical_density_capacity_cmy(profile))
                for profile in (reference, *endpoint_profiles)
            ]
        ),
        axis=0,
    )
    target_value = common_capacity * fixture["flat_density_fraction"]
    target = np.broadcast_to(target_value, (*shape, 3)).copy()
    high_target = np.broadcast_to(target_value, (*high_shape, 3)).copy()
    expected = np.power(10.0, -target_value)
    reference_rms = []
    for seed in fixture["development_seeds"]:
        rendered = block_aperture_mean(_render(reference, high_target, seed), factor)
        reference_rms.append(_rms(rendered - expected))
    evaluations: list[dict[str, Any]] = []

    def objective(multiplier: float) -> float:
        profile = compile_target_resolution_profile(
            reference, factor, rate_multiplier=float(multiplier)
        )
        losses = []
        per_seed_errors = []
        for seed, target_rms in zip(
            fixture["development_seeds"], reference_rms, strict=True
        ):
            candidate_rms = _rms(_render(profile, target, seed) - expected)
            log_ratio = np.log(candidate_rms / target_rms)
            losses.append(float(np.mean(np.square(log_ratio))))
            per_seed_errors.append(
                float(np.max(np.abs(candidate_rms - target_rms) / target_rms))
            )
        loss = float(np.median(losses))
        evaluations.append(
            {
                "rate_multiplier": float(multiplier),
                "objective": loss,
                "maximum_development_rms_relative_error": max(per_seed_errors),
            }
        )
        return loss

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": compiler["xatol"], "maxiter": compiler["maximum_iterations"]},
    )
    multiplier = float(result.x)
    selected = compile_target_resolution_profile(
        reference, factor, rate_multiplier=multiplier
    )
    development_error = next(
        row["maximum_development_rms_relative_error"]
        for row in reversed(evaluations)
        if row["rate_multiplier"] == multiplier
    )
    confirmation = []
    for seed in fixture["confirmation_seeds"]:
        ref = block_aperture_mean(_render(reference, high_target, seed), factor)
        candidate = _render(selected, target, seed)
        confirmation.append({"seed": seed, **_comparison(ref, candidate, expected)})
    edge_target = np.empty((*shape, 3), dtype=np.float64)
    edge_target[:, : shape[1] // 2] = common_capacity * 0.2
    edge_target[:, shape[1] // 2 :] = common_capacity * 0.8
    high_edge = np.repeat(np.repeat(edge_target, factor, axis=0), factor, axis=1)
    ref_edge = block_aperture_mean(
        _render(reference, high_edge, fixture["development_seeds"][0]), factor
    )
    candidate_edge = _render(selected, edge_target, fixture["development_seeds"][0])
    repeat_edge = _render(selected, edge_target, fixture["development_seeds"][0])
    ref_over, ref_under = _edge_excursion(ref_edge)
    cand_over, cand_under = _edge_excursion(candidate_edge)
    worst = {
        "development_marginal_noise_rms_relative_error": development_error,
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
        "development": worst["development_marginal_noise_rms_relative_error"]
        <= gates["maximum_development_marginal_noise_rms_relative_error"],
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
        "repeat": bool(np.array_equal(candidate_edge, repeat_edge)),
        "product_disabled": selected.product_enabled is False,
        "optimizer": bool(result.success),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "reference_profile_identity": reference.identity(),
        "selected_rate_multiplier": multiplier,
        "selected_profile_identity": selected.identity(),
        "optimizer_evaluations": len(evaluations),
        "confirmation_worst": worst,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4du_target_resolution_cloud_two_moment_compiler_report.v4",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "development": evaluations,
        "confirmation": confirmation,
    }


__all__ = ["evaluate", "load_contract"]
