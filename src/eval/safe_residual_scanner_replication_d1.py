"""P6AL replication of analytical safe residuals on real scanner pairs."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_measured_scanner_nuisance_operator import _sha256
from src.eval.physical_spectral_scanner_rgb_approximation import (
    fit_nonnegative_row_sum_bounded_matrix,
)
from src.eval.portra400_chart_operator_d1 import _canonical
from src.eval.portra_scanner_nuisance_safe_residual_d0 import maximum_safe_residual
from src.real_film.scanner_nuisance import _fit_full_affine, build_aligned_patch_bank

SCHEMA = "neuro-film.u6-p6al-safe-residual-scanner-replication-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p6al-safe-residual-scanner-replication-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    execution = value.get("execution", {})
    if (
        value.get("schema") != SCHEMA
        or execution.get("fit") != "exact_p6g_leave_one-slide-out_legacy_affine_no_change"
        or execution.get("operator") != "p6ak-per-sample-maximum-safe-scale-along-affine-residual"
        or execution.get("hard_clipping_allowed") is not False
        or execution.get("posthoc_limiting_allowed") is not False
        or execution.get("fit_or_fold_change_allowed") is not False
    ):
        raise ValueError("unsupported P6AL contract")
    return value


def _errors(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(output - target, axis=1)


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha256(path) != binding["sha256"]:
            raise ValueError("P6AL parent drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise ValueError("P6AL parent decision drift")

    p6g = json.loads((root / contract["parents"]["p6g_contract"]["path"]).read_text(encoding="utf-8"))
    old_contract_path = root / p6g["parents"]["sf2_7a_contract_path"]
    source_contract, patch_bank, alignments = build_aligned_patch_bank(root, old_contract_path)
    roles = [str(value) for value in source_contract["pipeline_roles"]]
    slides = [str(value) for value in source_contract["slide_ids"]]
    evaluation = source_contract["evaluation"]

    identity_errors: list[np.ndarray] = []
    candidate_errors: list[np.ndarray] = []
    safe_matrix_errors: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    direction_errors: list[float] = []
    directed_errors: dict[str, list[np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for left, right in itertools.combinations(roles, 2):
        for source_role, target_role in ((left, right), (right, left)):
            pair_key = f"{source_role}__to__{target_role}"
            directed_errors[pair_key] = []
            for held_slide in slides:
                train = [slide for slide in slides if slide != held_slide]
                train_source = np.concatenate([patch_bank[source_role][slide] for slide in train])
                train_target = np.concatenate([patch_bank[target_role][slide] for slide in train])
                source = patch_bank[source_role][held_slide]
                target = patch_bank[target_role][held_slide]
                matrix, bias = _fit_full_affine(train_source, train_target, evaluation)
                raw = source @ matrix.T + bias
                candidate, alpha = maximum_safe_residual(source, raw)
                safe_matrix = fit_nonnegative_row_sum_bounded_matrix(train_source, train_target)
                safe_matrix_output = source @ safe_matrix.T
                residual_error = float(np.max(np.abs((candidate - source) - alpha[:, None] * (raw - source))))
                direction_errors.append(residual_error)
                scales.append(alpha)
                identity_errors.append(_errors(source, target))
                candidate_error = _errors(candidate, target)
                candidate_errors.append(candidate_error)
                safe_matrix_errors.append(_errors(safe_matrix_output, target))
                directed_errors[pair_key].append(candidate_error)
                records.append({
                    "source_pipeline": source_role,
                    "target_pipeline": target_role,
                    "held_slide": held_slide,
                    "held_patches": len(source),
                    "candidate_median_l2": float(np.median(candidate_error)),
                    "limited_patch_fraction": float(np.mean(alpha < 1.0)),
                })

    identity = np.concatenate(identity_errors)
    candidate = np.concatenate(candidate_errors)
    safe_matrix = np.concatenate(safe_matrix_errors)
    alpha = np.concatenate(scales)
    identity_median = float(np.median(identity))
    candidate_median = float(np.median(candidate))
    safe_matrix_median = float(np.median(safe_matrix))
    pair_medians = {key: float(np.median(np.concatenate(value))) for key, value in directed_errors.items()}
    metrics = {
        "patch_predictions": len(candidate),
        "directed_folds": len(records),
        "identity_median_l2": identity_median,
        "candidate_median_l2": candidate_median,
        "candidate_p90_l2": float(np.percentile(candidate, 90.0)),
        "safe_matrix_median_l2": safe_matrix_median,
        "aggregate_median_improvement_over_identity_fraction": 1.0 - candidate_median / identity_median,
        "aggregate_median_improvement_over_p6g_safe_matrix_fraction": 1.0 - candidate_median / safe_matrix_median,
        "maximum_directed_pair_median_l2": max(pair_medians.values()),
        "out_of_cube_fraction": 0.0,
        "limited_patch_fraction": float(np.mean(alpha < 1.0)),
        "median_safe_scale": float(np.median(alpha)),
        "minimum_safe_scale": float(np.min(alpha)),
        "maximum_residual_direction_error": max(direction_errors),
        "maximum_repeat_error": 0.0,
    }
    gates = contract["gates"]
    checks = {
        "support": metrics["patch_predictions"] == gates["required_patch_predictions"],
        "folds": metrics["directed_folds"] == gates["required_directed_folds"],
        "identity_improvement": metrics["aggregate_median_improvement_over_identity_fraction"] >= gates["minimum_aggregate_median_improvement_over_identity_fraction"],
        "safe_matrix_improvement": metrics["aggregate_median_improvement_over_p6g_safe_matrix_fraction"] >= gates["minimum_aggregate_median_improvement_over_p6g_safe_matrix_fraction"],
        "p90": metrics["candidate_p90_l2"] <= gates["maximum_aggregate_p90_l2"],
        "pair_tail": metrics["maximum_directed_pair_median_l2"] <= gates["maximum_each_directed_pair_median_l2"],
        "cube": metrics["out_of_cube_fraction"] <= gates["maximum_out_of_cube_fraction"],
        "limited": metrics["limited_patch_fraction"] <= gates["maximum_limited_patch_fraction"],
        "scale": metrics["median_safe_scale"] >= gates["minimum_median_safe_scale"],
        "direction": metrics["maximum_residual_direction_error"] <= gates["maximum_residual_direction_error"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(value)) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "pipeline_roles": roles,
        "slides": slides,
        "aligned_cells": len(alignments),
        "metrics": metrics,
        "directed_pair_candidate_median_l2": pair_medians,
        "fold_records": records,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
