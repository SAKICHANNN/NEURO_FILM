"""BO7 grouped evaluation for the mature safe Bernstein LUT baseline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.flickr_luma_conditioned_matrix import _load_exact, _summary
from src.eval.flickr_single_author_pair_acquisition import canonical_bytes
from src.eval.flickr_weak_pair_operator_development import _load_rgb, _registered_samples, _rmse
from src.roll2film.factorized_monotone_bernstein import fit_factorized_monotone_bernstein
from src.roll2film.monotone_curve_matrix import fit_monotone_curve_positive_matrix


SCHEMA = "neuro-film.u5-r2bo7-flickr-safe-bernstein-lut.v1"


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise ValueError("invalid BO7 contract")
    decision = _load_exact(root, config["parents"]["bo6_decision"])
    if decision.get("decision") != config["parents"]["bo6_decision"]["required_decision"]:
        raise ValueError("BO6 decision drift")
    registration = _load_exact(root, config["parents"]["registration_report"])
    manifest = _load_exact(root, config["parents"]["download_manifest"])
    if registration.get("stable_evidence_id") != config["parents"]["registration_report"]["required_stable_evidence_id"]:
        raise ValueError("registration identity drift")
    return registration, manifest


def _fit_candidate(source: np.ndarray, target: np.ndarray, config: Mapping[str, Any], fold: int):
    candidate = config["candidate"]
    maximum = int(candidate["maximum_fit_samples"])
    if len(source) > maximum:
        indices = np.linspace(0, len(source) - 1, maximum, dtype=np.int64)
        source, target = source[indices], target[indices]
    return fit_factorized_monotone_bernstein(
        source,
        target,
        segment_count=int(candidate["segment_count"]),
        curve_learned_mixture=float(candidate["curve_learned_mixture"]),
        matrix_identity_mixture=float(candidate["matrix_identity_mixture"]),
        free_logit_bounds=tuple(float(value) for value in candidate["free_logit_bounds"]),
        restart_count=int(candidate["restart_count"]),
        maximum_function_evaluations=int(candidate["maximum_function_evaluations"]),
        function_tolerance=float(candidate["function_tolerance"]),
        parameter_tolerance=float(candidate["parameter_tolerance"]),
        gradient_tolerance=float(candidate["gradient_tolerance"]),
        loss=str(candidate["loss"]),
        loss_scale=float(candidate["loss_scale"]),
        seed=int(candidate["seed"]) + fold,
        residual_degree=int(candidate["residual_degree"]),
        residual_identity_ridge=float(candidate["residual_identity_ridge"]),
        jacobian_floor=float(candidate["jacobian_floor"]),
        safety_grid_size=int(candidate["safety_grid_size"]),
        strength_steps=int(candidate["strength_steps"]),
        maximum_residual_iterations=int(candidate["maximum_residual_iterations"]),
    )


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registration, manifest = validate_inputs(root, config)
    manifest_rows = {str(row["local_path"]): row for row in manifest["rows"]}
    accepted = [row for row in registration["pairs"] if row["diagnostics"]["registration_gate_passed"]]
    development = [row for row in accepted if int(row["scene_id"]) % 4 != 0]
    confirmation = [row for row in accepted if int(row["scene_id"]) % 4 == 0]
    data_root = root / str(config["data_root"])
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in development:
        digital_meta = manifest_rows[str(row["digital_local_path"])]
        film_meta = manifest_rows[str(row["film_local_path"])]
        samples[str(row["pair_id"])] = _registered_samples(
            _load_rgb(data_root / str(row["digital_local_path"]), str(digital_meta["sha256"])),
            _load_rgb(data_root / str(row["film_local_path"]), str(film_meta["sha256"])),
            np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["paired_samples"],
        )[:2]

    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    families = sorted({str(row["family_id"]) for row in development})
    axis = np.linspace(0.0, 1.0, int(config["evaluation"]["dense_cube_grid_size"]), dtype=np.float64)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    epsilon = float(config["evaluation"]["strict_boundary_epsilon"])
    source_boundary = np.any((cube <= epsilon) | (cube >= 1.0 - epsilon), axis=1)
    for fold in range(3):
        for family in families:
            train = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 != fold]
            test = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 == fold]
            train_source = np.concatenate([samples[str(row["pair_id"])][0] for row in train])
            train_target = np.concatenate([samples[str(row["pair_id"])][1] for row in train])
            candidate = _fit_candidate(train_source, train_target, config, fold)
            control_cfg = config["control"]
            control = fit_monotone_curve_positive_matrix(
                train_source,
                train_target,
                curve_identity_mixture=float(control_cfg["curve_identity_mixture"]),
                matrix_identity_mixture=float(control_cfg["matrix_identity_mixture"]),
                free_logit_bounds=tuple(float(value) for value in control_cfg["free_logit_bounds"]),
                restart_count=int(control_cfg["restart_count"]),
                maximum_function_evaluations=int(control_cfg["maximum_fit_evaluations"]),
                loss=str(control_cfg["loss"]),
                loss_scale=float(control_cfg["loss_scale"]),
                seed=int(control_cfg["seed"]) + fold,
            )
            cube_output = candidate.operator.apply(cube)
            output_boundary = np.any((cube_output <= epsilon) | (cube_output >= 1.0 - epsilon), axis=1)
            combined_jacobian = candidate.operator.jacobian_determinants(cube)
            fits.append({
                "fold": fold,
                "family_id": family,
                "candidate_converged": candidate.converged,
                "control_converged": control.converged,
                "safe_residual_strength": candidate.operator.residual.strength,
                "minimum_combined_jacobian_determinant": float(np.min(combined_jacobian)),
                "new_boundary_fraction": float(np.mean(output_boundary & ~source_boundary)),
                "operator": candidate.operator.to_dict(),
            })
            for row in test:
                source, target = samples[str(row["pair_id"])]
                candidate_output = candidate.operator.apply(source)
                control_output = control.operator.apply(source)
                rows.append({
                    "pair_id": row["pair_id"],
                    "family_id": family,
                    "scene_id": row["scene_id"],
                    "fold": fold,
                    "errors": {"control": _rmse(control_output, target), "candidate": _rmse(candidate_output, target)},
                })

    summaries = {method: _summary(rows, method) for method in ("control", "candidate")}
    control_errors = np.asarray([row["errors"]["control"] for row in rows], dtype=np.float64)
    candidate_errors = np.asarray([row["errors"]["candidate"] for row in rows], dtype=np.float64)
    improvements = (control_errors - candidate_errors) / np.maximum(control_errors, 1e-20)
    family_improvements = {
        family: float(np.median([improvements[index] for index, row in enumerate(rows) if row["family_id"] == family]))
        for family in families
    }
    metrics = {
        "development_scenes": len(rows),
        "sealed_confirmation_scenes_not_loaded": len(confirmation),
        "summaries": summaries,
        "mean_improvement_over_control": float((summaries["control"]["mean_rmse"] - summaries["candidate"]["mean_rmse"]) / summaries["control"]["mean_rmse"]),
        "median_improvement_over_control": float(np.median(improvements)),
        "scene_win_fraction_over_control": float(np.mean(candidate_errors < control_errors)),
        "per_family_median_improvement_over_control": family_improvements,
        "p95_error_ratio_to_control": float(summaries["candidate"]["p95_rmse"] / summaries["control"]["p95_rmse"]),
        "worst_error_ratio_to_control": float(summaries["candidate"]["worst_rmse"] / summaries["control"]["worst_rmse"]),
        "minimum_safe_residual_strength": float(min(fit["safe_residual_strength"] for fit in fits)),
        "minimum_combined_jacobian_determinant": float(min(fit["minimum_combined_jacobian_determinant"] for fit in fits)),
        "maximum_new_boundary_fraction": float(max(fit["new_boundary_fraction"] for fit in fits)),
        "all_fits_converged": all(fit["candidate_converged"] and fit["control_converged"] for fit in fits),
    }
    gates = config["evaluation"]
    checks = {
        "mean_improvement": metrics["mean_improvement_over_control"] >= float(gates["minimum_candidate_mean_improvement_over_control"]),
        "median_improvement": metrics["median_improvement_over_control"] >= float(gates["minimum_candidate_median_improvement_over_control"]),
        "scene_win_fraction": metrics["scene_win_fraction_over_control"] >= float(gates["minimum_candidate_scene_win_fraction_over_control"]),
        "every_family_nonnegative": all(value >= float(gates["minimum_each_family_median_improvement_over_control"]) for value in family_improvements.values()),
        "p95_tail": metrics["p95_error_ratio_to_control"] <= float(gates["maximum_candidate_p95_error_ratio_to_control"]),
        "worst_tail": metrics["worst_error_ratio_to_control"] <= float(gates["maximum_candidate_worst_error_ratio_to_control"]),
        "residual_strength": metrics["minimum_safe_residual_strength"] >= float(gates["minimum_safe_residual_strength"]),
        "jacobian": metrics["minimum_combined_jacobian_determinant"] >= float(gates["minimum_combined_jacobian_determinant"]),
        "new_boundary": metrics["maximum_new_boundary_fraction"] <= float(gates["maximum_new_boundary_fraction"]),
        "fit_convergence": metrics["all_fits_converged"] is bool(gates["require_all_fits_converged"]),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bo7-flickr-safe-bernstein-lut-report.v1",
        "node": config["node"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "selected_candidate": "factorized_safe_bernstein_lut" if automatic_pass else None,
        "branch": config["branches"]["pass" if automatic_pass else "fail"],
        "fits": fits,
        "rows": rows,
        "confirmation_pixels_loaded": False,
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}

