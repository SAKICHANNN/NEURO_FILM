"""U6.P6G bounded scanner-nuisance operators on same-slide real scans."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from src.eval.physical_spectral_scanner_rgb_approximation import (
    fit_nonnegative_row_sum_bounded_matrix,
)
from src.real_film.scanner_nuisance import (
    _fit_full_affine,
    build_aligned_patch_bank,
)


SCHEMA = "neuro_film.u6_p6g_measured_scanner_nuisance_operator_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6G contract")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_l2": float(np.mean(errors)),
        "median_l2": float(np.median(errors)),
        "p90_l2": float(np.percentile(errors, 90.0)),
        "p95_l2": float(np.percentile(errors, 95.0)),
        "maximum_l2": float(np.max(errors)),
    }


def _output_diagnostics(values: np.ndarray) -> dict[str, float | int]:
    outside = (values < 0.0) | (values > 1.0)
    row_outside = np.any(outside, axis=1)
    excursion = np.maximum(-values, values - 1.0)
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "outside_scalar_count": int(np.sum(outside)),
        "outside_scalar_fraction": float(np.mean(outside)),
        "outside_patch_count": int(np.sum(row_outside)),
        "outside_patch_fraction": float(np.mean(row_outside)),
        "maximum_boundary_excursion": float(np.max(np.maximum(excursion, 0.0))),
    }


def _merge_output_diagnostics(chunks: list[np.ndarray]) -> dict[str, float | int]:
    return _output_diagnostics(np.concatenate(chunks, axis=0))


def evaluate_measured_scanner_nuisance_operator(
    root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = load_contract(contract_path)
    parents = contract["parents"]
    parent_pairs = (
        ("sf2_7r_contract_path", "sf2_7r_contract_sha256"),
        ("sf2_7r_audit_path", "sf2_7r_audit_sha256"),
        ("sf2_7a_contract_path", "sf2_7a_contract_sha256"),
        ("sf2_7a_report_path", "sf2_7a_report_sha256"),
        ("sf2_7a_decision_path", "sf2_7a_decision_sha256"),
    )
    for path_key, hash_key in parent_pairs:
        if _sha256(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"{path_key} hash mismatch")

    source_audit = json.loads(
        (root / parents["sf2_7r_audit_path"]).read_text(encoding="utf-8")
    )
    old_report = json.loads(
        (root / parents["sf2_7a_report_path"]).read_text(encoding="utf-8")
    )
    old_decision = json.loads(
        (root / parents["sf2_7a_decision_path"]).read_text(encoding="utf-8")
    )
    if not source_audit.get("all_checks_passed"):
        raise ValueError("SF2.7R source audit is not passing")
    if (
        old_decision.get("decision") != "partial_control_pass_no_stock_permission"
        or old_report.get("all_alignment_gates_passed") is not True
    ):
        raise ValueError("SF2.7A parent decision is not exact")

    old_contract_path = root / parents["sf2_7a_contract_path"]
    source_contract, patch_bank, alignments = build_aligned_patch_bank(
        root, old_contract_path
    )
    evidence = contract["evidence_contract"]
    roles = [str(value) for value in source_contract["pipeline_roles"]]
    slides = [str(value) for value in source_contract["slide_ids"]]
    expected_shape = (int(evidence["patches_per_slide"]), 3)
    if len(roles) != int(evidence["scanner_software_pipelines"]):
        raise ValueError("scanner pipeline count mismatch")
    if len(slides) != int(evidence["slides"]):
        raise ValueError("slide count mismatch")
    if len(alignments) != int(evidence["aligned_pipeline_slide_cells"]):
        raise ValueError("aligned cell count mismatch")
    if any(
        patch_bank[role][slide].shape != expected_shape
        for role in roles
        for slide in slides
    ):
        raise ValueError("aligned patch-bank shape mismatch")

    evaluation = source_contract["evaluation"]
    model_outputs: dict[str, list[np.ndarray]] = {
        "identity": [],
        "legacy-unclipped-full-affine-diagnostic-only": [],
        "legacy-clipped-full-affine-parent-replay": [],
        "nonnegative-row-sum-bounded-3x3": [],
    }
    model_errors: dict[str, list[np.ndarray]] = {
        key: [] for key in model_outputs
    }
    directed_pair_errors: dict[str, list[np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    safe_matrices: list[np.ndarray] = []
    legacy_matrices: list[np.ndarray] = []
    legacy_biases: list[np.ndarray] = []

    for left, right in itertools.combinations(roles, 2):
        for source_role, target_role in ((left, right), (right, left)):
            pair_key = f"{source_role}__to__{target_role}"
            directed_pair_errors[pair_key] = []
            for held_slide in slides:
                train_slides = [
                    slide for slide in slides if slide != held_slide
                ]
                train_source = np.concatenate(
                    [patch_bank[source_role][slide] for slide in train_slides]
                )
                train_target = np.concatenate(
                    [patch_bank[target_role][slide] for slide in train_slides]
                )
                source = patch_bank[source_role][held_slide]
                target = patch_bank[target_role][held_slide]
                legacy_matrix, legacy_bias = _fit_full_affine(
                    train_source, train_target, evaluation
                )
                safe_matrix = fit_nonnegative_row_sum_bounded_matrix(
                    train_source, train_target
                )
                outputs = {
                    "identity": source,
                    "legacy-unclipped-full-affine-diagnostic-only": (
                        source @ legacy_matrix.T + legacy_bias
                    ),
                    "legacy-clipped-full-affine-parent-replay": np.clip(
                        source @ legacy_matrix.T + legacy_bias, 0.0, 1.0
                    ),
                    "nonnegative-row-sum-bounded-3x3": source @ safe_matrix.T,
                }
                per_model: dict[str, Any] = {}
                for model, output in outputs.items():
                    error = np.linalg.norm(output - target, axis=1)
                    model_outputs[model].append(output)
                    model_errors[model].append(error)
                    per_model[model] = {
                        **_summary(error),
                        "output": _output_diagnostics(output),
                    }
                directed_pair_errors[pair_key].append(
                    np.linalg.norm(
                        outputs["nonnegative-row-sum-bounded-3x3"] - target,
                        axis=1,
                    )
                )
                safe_matrices.append(safe_matrix)
                legacy_matrices.append(legacy_matrix)
                legacy_biases.append(legacy_bias)
                records.append(
                    {
                        "source_pipeline": source_role,
                        "target_pipeline": target_role,
                        "held_slide": held_slide,
                        "models": per_model,
                        "safe_matrix": safe_matrix.tolist(),
                        "legacy_matrix": legacy_matrix.tolist(),
                        "legacy_bias": legacy_bias.tolist(),
                    }
                )

    aggregate = {
        model: {
            **_summary(np.concatenate(chunks)),
            "output": _merge_output_diagnostics(model_outputs[model]),
            "patch_predictions": int(sum(chunk.size for chunk in chunks)),
        }
        for model, chunks in model_errors.items()
    }
    pair_summaries = {
        pair: _summary(np.concatenate(chunks))
        for pair, chunks in directed_pair_errors.items()
    }
    safe_matrix_stack = np.stack(safe_matrices)
    legacy_matrix_stack = np.stack(legacy_matrices)
    legacy_bias_stack = np.stack(legacy_biases)
    fit_diagnostics = {
        "safe_matrix_minimum_coefficient": float(np.min(safe_matrix_stack)),
        "safe_matrix_maximum_row_sum": float(
            np.max(np.sum(safe_matrix_stack, axis=2))
        ),
        "legacy_negative_coefficient_count": int(
            np.sum(legacy_matrix_stack < 0.0)
        ),
        "legacy_negative_bias_count": int(np.sum(legacy_bias_stack < 0.0)),
        "legacy_positive_bias_count": int(np.sum(legacy_bias_stack > 0.0)),
        "fold_direction_fits": int(safe_matrix_stack.shape[0]),
    }

    old_aggregate = old_report["aggregate"]
    replay_checks = {
        "identity_median_matches_parent": bool(
            np.isclose(
                aggregate["identity"]["median_l2"],
                old_aggregate["identity"]["median_rgb_euclidean"],
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "legacy_clipped_median_matches_parent": bool(
            np.isclose(
                aggregate["legacy-clipped-full-affine-parent-replay"][
                    "median_l2"
                ],
                old_aggregate["bounded_full_affine_3x3_plus_bias"][
                    "median_rgb_euclidean"
                ],
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "legacy_clipped_p90_matches_parent": bool(
            np.isclose(
                aggregate["legacy-clipped-full-affine-parent-replay"]["p90_l2"],
                old_aggregate["bounded_full_affine_3x3_plus_bias"][
                    "p90_rgb_euclidean"
                ],
                rtol=0.0,
                atol=1e-15,
            )
        ),
    }
    identity_median = aggregate["identity"]["median_l2"]
    safe_median = aggregate["nonnegative-row-sum-bounded-3x3"]["median_l2"]
    improvement = (
        (identity_median - safe_median) / identity_median
        if identity_median > 0.0
        else 0.0
    )
    gates = contract["automatic_gates"]
    safe_output = aggregate["nonnegative-row-sum-bounded-3x3"]["output"]
    checks = {
        "parent_source_pass": bool(source_audit["all_checks_passed"]),
        "all_alignment_gates_pass": len(alignments)
        == int(gates["aligned_pipeline_slide_cells"]),
        "parent_replay_exact": all(replay_checks.values()),
        "legacy_unclipped_factual_diagnostics_present": bool(
            fit_diagnostics["legacy_negative_coefficient_count"] >= 0
            and aggregate[
                "legacy-unclipped-full-affine-diagnostic-only"
            ]["output"]["outside_scalar_count"]
            >= 0
        ),
        "safe_matrix_structure": bool(
            fit_diagnostics["safe_matrix_minimum_coefficient"]
            >= float(gates["safe_matrix_coefficient_minimum"]) - 1e-15
            and fit_diagnostics["safe_matrix_maximum_row_sum"]
            <= float(gates["safe_matrix_row_sum_maximum"]) + 1e-15
        ),
        "safe_output_bounded": bool(
            safe_output["minimum"] >= float(gates["safe_output_minimum"]) - 1e-15
            and safe_output["maximum"]
            <= float(gates["safe_output_maximum"]) + 1e-15
            and safe_output["outside_scalar_count"] == 0
        ),
        "safe_relative_improvement": improvement
        >= float(
            gates["safe_matrix_vs_identity_aggregate_median_improvement_min"]
        ),
        "safe_aggregate_median": safe_median
        <= float(gates["safe_matrix_aggregate_median_l2_max"]),
        "safe_aggregate_p90": aggregate[
            "nonnegative-row-sum-bounded-3x3"
        ]["p90_l2"]
        <= float(gates["safe_matrix_aggregate_p90_l2_max"]),
        "safe_each_directed_pair_median": all(
            row["median_l2"]
            <= float(gates["safe_matrix_each_directed_pair_median_l2_max"])
            for row in pair_summaries.values()
        ),
    }

    stable_payload = {
        "schema": "neuro_film.u6_p6g_measured_scanner_nuisance_operator_report.v1",
        "node": contract["node"],
        "config_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "parents_verified": {
            hash_key: parents[hash_key] for _, hash_key in parent_pairs
        },
        "evidence": {
            "pipeline_roles": roles,
            "slides": slides,
            "aligned_cells": len(alignments),
            "patches_per_slide": expected_shape[0],
            "colour_state": source_contract["decode"]["colour_state"],
        },
        "alignment_summary": {
            "minimum_ransac_inliers": min(
                int(row["ransac_inliers"]) for row in alignments
            ),
            "maximum_median_inlier_reprojection_error_px": max(
                float(row["median_inlier_reprojection_error_px"])
                for row in alignments
            ),
        },
        "aggregate": aggregate,
        "directed_pair_safe_matrix": pair_summaries,
        "fit_diagnostics": fit_diagnostics,
        "parent_replay_checks": replay_checks,
        "safe_matrix_median_improvement_fraction": improvement,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": contract["claim_ceiling"],
        "forbidden_claims": contract["forbidden_claims"],
    }
    return {
        **stable_payload,
        "stable_evidence_id": _stable_id(stable_payload),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "directed_fold_records": records,
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "evaluate_measured_scanner_nuisance_operator",
    "load_contract",
    "write_report",
]
