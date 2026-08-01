"""BO6 held-scene luma-conditioned positive mixing evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file
from src.eval.flickr_weak_pair_operator_development import _fit_basic, _load_rgb, _registered_samples, _rmse
from src.roll2film.luma_conditioned_positive_matrix import fit_luma_conditioned_positive_matrix
from src.roll2film.monotone_curve_matrix import fit_monotone_curve_positive_matrix


SCHEMA = "neuro-film.u5-r2bo6-flickr-luma-conditioned-matrix.v1"


class FlickrLumaConditionedMatrixError(ValueError):
    pass


def _load_exact(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise FlickrLumaConditionedMatrixError(f"parent hash drift: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrLumaConditionedMatrixError("invalid BO6 contract")
    for key in ("bo3_decision", "bo5_decision"):
        decision = _load_exact(root, config["parents"][key])
        if decision.get("decision") != config["parents"][key]["required_decision"]:
            raise FlickrLumaConditionedMatrixError(f"required decision drift: {key}")
    registration = _load_exact(root, config["parents"]["registration_report"])
    manifest = _load_exact(root, config["parents"]["download_manifest"])
    if registration.get("stable_evidence_id") != config["parents"]["registration_report"]["required_stable_evidence_id"]:
        raise FlickrLumaConditionedMatrixError("registration identity drift")
    return registration, manifest


def _summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    errors = np.asarray([row["errors"][method] for row in rows], dtype=np.float64)
    return {
        "mean_rmse": float(np.mean(errors)),
        "median_rmse": float(np.median(errors)),
        "p95_rmse": float(np.quantile(errors, 0.95)),
        "worst_rmse": float(np.max(errors)),
    }


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
    for fold in range(3):
        for family in families:
            train = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 != fold]
            test = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 == fold]
            train_source = np.concatenate([samples[str(row["pair_id"])][0] for row in train])
            train_target = np.concatenate([samples[str(row["pair_id"])][1] for row in train])
            basic, basic_ok = _fit_basic(train_source, train_target, {"operators": {"family_basic_logit": {
                "log_scale_bounds": [-0.7, 0.7], "shift_bounds": [-1.0, 1.0], "identity_shrinkage": 0.05, "maximum_fit_evaluations": 100
            }}})
            basic_train = basic.apply(train_source)
            candidate_cfg = config["candidate"]
            candidate, candidate_ok = fit_luma_conditioned_positive_matrix(
                basic_train,
                train_target,
                maximum_off_diagonal=float(candidate_cfg["off_diagonal_maximum_before_row_normalization"]),
                logit_bounds=tuple(float(value) for value in candidate_cfg["off_diagonal_logit_bounds"]),
                initial_logit=float(candidate_cfg["initial_off_diagonal_logit"]),
                identity_shrinkage=float(candidate_cfg["identity_shrinkage"]),
                maximum_fit_samples=int(candidate_cfg["maximum_fit_samples"]),
                maximum_evaluations=int(candidate_cfg["maximum_fit_evaluations"]),
                loss=str(candidate_cfg["loss"]),
                loss_scale=float(candidate_cfg["loss_scale"]),
            )
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
            fits.append({
                "fold": fold,
                "family_id": family,
                "basic_converged": basic_ok,
                "candidate_converged": candidate_ok,
                "control_converged": control.converged,
                "candidate_matrices": candidate.matrices.tolist(),
                "candidate_parameters": candidate.parameters.tolist(),
            })
            for row in test:
                source, target = samples[str(row["pair_id"])]
                basic_output = basic.apply(source)
                candidate_output = candidate.apply(basic_output, conditioning_rgb=basic_output)
                control_output = control.operator.apply(source)
                rows.append({
                    "pair_id": row["pair_id"],
                    "family_id": family,
                    "scene_id": row["scene_id"],
                    "fold": fold,
                    "errors": {
                        "basic": _rmse(basic_output, target),
                        "control": _rmse(control_output, target),
                        "candidate": _rmse(candidate_output, target),
                    },
                })

    summaries = {method: _summary(rows, method) for method in ("basic", "control", "candidate")}
    control_errors = np.asarray([row["errors"]["control"] for row in rows], dtype=np.float64)
    candidate_errors = np.asarray([row["errors"]["candidate"] for row in rows], dtype=np.float64)
    improvements = (control_errors - candidate_errors) / np.maximum(control_errors, 1e-20)
    family_improvements = {
        family: float(np.median([improvements[index] for index, row in enumerate(rows) if row["family_id"] == family]))
        for family in families
    }
    grid_size = int(config["evaluation"]["dense_cube_grid_size"])
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    epsilon = float(config["evaluation"]["strict_boundary_epsilon"])
    source_boundary = np.any((cube <= epsilon) | (cube >= 1.0 - epsilon), axis=1)
    new_boundary_fractions = []
    for fit in fits:
        from src.roll2film.luma_conditioned_positive_matrix import LumaConditionedPositiveMatrix
        operator = LumaConditionedPositiveMatrix(
            np.asarray(fit["candidate_parameters"], dtype=np.float64),
            float(config["candidate"]["off_diagonal_maximum_before_row_normalization"]),
        )
        output = operator.apply(cube)
        output_boundary = np.any((output <= epsilon) | (output >= 1.0 - epsilon), axis=1)
        new_boundary_fractions.append(float(np.mean(output_boundary & ~source_boundary)))
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
        "maximum_new_boundary_fraction": max(new_boundary_fractions),
        "all_fits_converged": all(fit["basic_converged"] and fit["candidate_converged"] and fit["control_converged"] for fit in fits),
    }
    gates = config["evaluation"]
    checks = {
        "mean_improvement": metrics["mean_improvement_over_control"] >= float(gates["minimum_candidate_mean_improvement_over_control"]),
        "median_improvement": metrics["median_improvement_over_control"] >= float(gates["minimum_candidate_median_improvement_over_control"]),
        "scene_win_fraction": metrics["scene_win_fraction_over_control"] >= float(gates["minimum_candidate_scene_win_fraction_over_control"]),
        "every_family_nonnegative": all(value >= float(gates["minimum_each_family_median_improvement_over_control"]) for value in family_improvements.values()),
        "p95_tail": metrics["p95_error_ratio_to_control"] <= float(gates["maximum_candidate_p95_error_ratio_to_control"]),
        "worst_tail": metrics["worst_error_ratio_to_control"] <= float(gates["maximum_candidate_worst_error_ratio_to_control"]),
        "new_boundary": metrics["maximum_new_boundary_fraction"] <= float(gates["maximum_new_boundary_fraction"]),
        "fit_convergence": metrics["all_fits_converged"] is bool(gates["require_all_fits_converged"]),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bo6-flickr-luma-conditioned-matrix-report.v1",
        "node": config["node"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "selected_candidate": "luma_conditioned_positive_matrix" if automatic_pass else None,
        "branch": config["branches"]["pass" if automatic_pass else "fail"],
        "fits": fits,
        "rows": rows,
        "confirmation_pixels_loaded": False,
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}

