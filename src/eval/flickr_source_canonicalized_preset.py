"""BO8 source-only canonicalizer plus shared explicit preset audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.eval.flickr_luma_conditioned_matrix import _load_exact, _summary
from src.eval.flickr_single_author_pair_acquisition import canonical_bytes
from src.eval.flickr_weak_pair_operator_development import _load_rgb, _registered_samples, _rmse
from src.roll2film.monotone_curve_matrix import fit_monotone_curve_positive_matrix
from src.roll2film.source_logit_canonicalizer import estimate_source_logit_shift, logit_shift


SCHEMA = "neuro-film.u5-r2bo8-flickr-source-canonicalized-preset.v1"


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise ValueError("invalid BO8 contract")
    decision = _load_exact(root, config["parents"]["bo7_decision"])
    if decision.get("decision") != config["parents"]["bo7_decision"]["required_decision"]:
        raise ValueError("BO7 decision drift")
    registration = _load_exact(root, config["parents"]["registration_report"])
    manifest = _load_exact(root, config["parents"]["download_manifest"])
    if registration.get("stable_evidence_id") != config["parents"]["registration_report"]["required_stable_evidence_id"]:
        raise ValueError("registration identity drift")
    return registration, manifest


def _source_only_shift(digital_u8: np.ndarray, homography: np.ndarray, target_shape: tuple[int, int], config: Mapping[str, Any]) -> np.ndarray:
    height, width = target_shape
    encoded = digital_u8.astype(np.float64) / 255.0
    warped = cv2.warpPerspective(encoded, homography, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.ones(digital_u8.shape[:2], dtype=np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
    erosion = int(config["mask_erosion_pixels"])
    if erosion:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((erosion, erosion), np.uint8)).astype(bool)
    low = float(config["encoded_boundary_code_minimum"]) / 255.0
    high = float(config["encoded_boundary_code_maximum"]) / 255.0
    valid &= np.all((warped > low) & (warped < high), axis=2)
    indices = np.flatnonzero(valid)
    if len(indices) < 256:
        raise ValueError("insufficient source-only canonicalizer support")
    maximum = int(config["maximum_statistic_samples"])
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]
    return estimate_source_logit_shift(
        warped.reshape(-1, 3)[indices],
        maximum_absolute_shift=float(config["maximum_absolute_logit_shift"]),
    )


def _fit(source: np.ndarray, target: np.ndarray, config: Mapping[str, Any], fold: int):
    return fit_monotone_curve_positive_matrix(
        source,
        target,
        curve_identity_mixture=float(config["curve_identity_mixture"]),
        matrix_identity_mixture=float(config["matrix_identity_mixture"]),
        free_logit_bounds=tuple(float(value) for value in config["free_logit_bounds"]),
        restart_count=int(config["restart_count"]),
        maximum_function_evaluations=int(config["maximum_fit_evaluations"]),
        loss=str(config["loss"]),
        loss_scale=float(config["loss_scale"]),
        seed=int(config["seed"]) + fold,
    )


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registration, manifest = validate_inputs(root, config)
    manifest_rows = {str(row["local_path"]): row for row in manifest["rows"]}
    accepted = [row for row in registration["pairs"] if row["diagnostics"]["registration_gate_passed"]]
    development = [row for row in accepted if int(row["scene_id"]) % 4 != 0]
    confirmation = [row for row in accepted if int(row["scene_id"]) % 4 == 0]
    data_root = root / str(config["data_root"])
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shifts: dict[str, np.ndarray] = {}
    for row in development:
        pair_id = str(row["pair_id"])
        digital_meta = manifest_rows[str(row["digital_local_path"])]
        film_meta = manifest_rows[str(row["film_local_path"])]
        digital = _load_rgb(data_root / str(row["digital_local_path"]), str(digital_meta["sha256"]))
        film = _load_rgb(data_root / str(row["film_local_path"]), str(film_meta["sha256"]))
        homography = np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64)
        samples[pair_id] = _registered_samples(digital, film, homography, config["paired_samples"])[:2]
        shifts[pair_id] = _source_only_shift(digital, homography, film.shape[:2], config["source_adapter"])

    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    families = sorted({str(row["family_id"]) for row in development})
    wrong_shift: dict[str, np.ndarray] = {}
    for family in families:
        family_rows = sorted((row for row in development if row["family_id"] == family), key=lambda row: str(row["pair_id"]))
        for index, row in enumerate(family_rows):
            wrong_shift[str(row["pair_id"])] = shifts[str(family_rows[(index + 1) % len(family_rows)]["pair_id"])]
    epsilon = float(config["evaluation"]["strict_boundary_epsilon"])
    axis = np.linspace(0.0, 1.0, int(config["evaluation"]["dense_cube_grid_size"]), dtype=np.float64)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    source_boundary = np.any((cube <= epsilon) | (cube >= 1.0 - epsilon), axis=1)
    boundary_rows: list[float] = []
    for fold in range(3):
        for family in families:
            train = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 != fold]
            test = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 == fold]
            canonical_source = np.concatenate([logit_shift(samples[str(row["pair_id"])][0], shifts[str(row["pair_id"])]) for row in train])
            canonical_target = np.concatenate([logit_shift(samples[str(row["pair_id"])][1], shifts[str(row["pair_id"])]) for row in train])
            original_source = np.concatenate([samples[str(row["pair_id"])][0] for row in train])
            original_target = np.concatenate([samples[str(row["pair_id"])][1] for row in train])
            candidate = _fit(canonical_source, canonical_target, config["shared_preset"], fold)
            control = _fit(original_source, original_target, config["shared_preset"], fold)
            fits.append({"fold": fold, "family_id": family, "candidate_converged": candidate.converged, "control_converged": control.converged, "candidate_operator": candidate.operator.to_dict(), "control_operator": control.operator.to_dict()})
            for row in test:
                pair_id = str(row["pair_id"])
                source, target = samples[pair_id]
                shift = shifts[pair_id]
                candidate_output = logit_shift(candidate.operator.apply(logit_shift(source, shift)), -shift)
                wrong = wrong_shift[pair_id]
                wrong_output = logit_shift(candidate.operator.apply(logit_shift(source, wrong)), -wrong)
                control_output = control.operator.apply(source)
                cube_output = logit_shift(candidate.operator.apply(logit_shift(cube, shift)), -shift)
                output_boundary = np.any((cube_output <= epsilon) | (cube_output >= 1.0 - epsilon), axis=1)
                boundary_rows.append(float(np.mean(output_boundary & ~source_boundary)))
                rows.append({
                    "pair_id": pair_id,
                    "family_id": family,
                    "scene_id": row["scene_id"],
                    "fold": fold,
                    "source_only_logit_shift": shift.tolist(),
                    "wrong_adapter_pair_id": next(key for key, value in shifts.items() if value is wrong),
                    "errors": {"control": _rmse(control_output, target), "wrong_adapter": _rmse(wrong_output, target), "candidate": _rmse(candidate_output, target)},
                })

    summaries = {method: _summary(rows, method) for method in ("control", "wrong_adapter", "candidate")}
    control_errors = np.asarray([row["errors"]["control"] for row in rows])
    wrong_errors = np.asarray([row["errors"]["wrong_adapter"] for row in rows])
    candidate_errors = np.asarray([row["errors"]["candidate"] for row in rows])
    improvements = (control_errors - candidate_errors) / np.maximum(control_errors, 1e-20)
    wrong_improvements = (wrong_errors - candidate_errors) / np.maximum(wrong_errors, 1e-20)
    family_improvements = {family: float(np.median([improvements[index] for index, row in enumerate(rows) if row["family_id"] == family])) for family in families}
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
        "median_improvement_over_wrong_adapter": float(np.median(wrong_improvements)),
        "maximum_observed_absolute_logit_shift": float(max(np.max(np.abs(value)) for value in shifts.values())),
        "maximum_new_boundary_fraction": max(boundary_rows),
        "all_fits_converged": all(row["candidate_converged"] and row["control_converged"] for row in fits),
    }
    gates = config["evaluation"]
    checks = {
        "mean_improvement": metrics["mean_improvement_over_control"] >= float(gates["minimum_candidate_mean_improvement_over_uncanonicalized_control"]),
        "median_improvement": metrics["median_improvement_over_control"] >= float(gates["minimum_candidate_median_improvement_over_uncanonicalized_control"]),
        "scene_win_fraction": metrics["scene_win_fraction_over_control"] >= float(gates["minimum_candidate_scene_win_fraction_over_uncanonicalized_control"]),
        "every_family_nonnegative": all(value >= float(gates["minimum_each_family_median_improvement_over_uncanonicalized_control"]) for value in family_improvements.values()),
        "p95_tail": metrics["p95_error_ratio_to_control"] <= float(gates["maximum_candidate_p95_error_ratio_to_uncanonicalized_control"]),
        "worst_tail": metrics["worst_error_ratio_to_control"] <= float(gates["maximum_candidate_worst_error_ratio_to_uncanonicalized_control"]),
        "wrong_adapter_discrimination": metrics["median_improvement_over_wrong_adapter"] >= float(gates["minimum_candidate_median_improvement_over_wrong_adapter"]),
        "adapter_bound": metrics["maximum_observed_absolute_logit_shift"] <= float(gates["maximum_observed_absolute_logit_shift"]),
        "new_boundary": metrics["maximum_new_boundary_fraction"] <= float(gates["maximum_new_boundary_fraction"]),
        "fit_convergence": metrics["all_fits_converged"] is bool(gates["require_all_fits_converged"]),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bo8-flickr-source-canonicalized-preset-report.v1",
        "node": config["node"],
        "information_flow": {"source_adapter_uses_application_source": True, "source_adapter_uses_target_or_reference": False, "shared_preset_uses_paired_development_only": True, "confirmation_pixels_loaded": False},
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "selected_candidate": "source_logit_canonicalizer_plus_shared_preset" if automatic_pass else None,
        "branch": config["branches"]["pass" if automatic_pass else "fail"],
        "fits": fits,
        "rows": rows,
        "confirmation_pixels_loaded": False,
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}

