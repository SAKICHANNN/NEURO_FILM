#!/usr/bin/env python3
"""Fit and score the frozen REPID shared global logit-affine operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_u5_r2spcp2_global_logit_affine_d0 import (
    _operator_payload,
    _relative_gain,
    _select_dose,
)
from src.eval.repid_shared_operator import load_jpeg_as_srgb
from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    fit_operator_rows,
    gradient_p999_ratio,
    mean_oklab_error,
    new_exact_boundary_fraction,
    sample_indexes,
    srgb_code_to_oklab,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lookup(acquisition: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (record["scene_id"], member["endpoint"]): member
        for record in acquisition["records"]
        for member in record["members"]
    }


def _load_pair(
    row: dict[str, Any], lookup: dict[tuple[str, str], dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for endpoint in ("loser", "winner"):
        member = lookup[(row["scene_id"], endpoint)]
        arrays.append(
            load_jpeg_as_srgb(
                ROOT / member["relative_path"],
                expected_sha256=member["sha256"],
                expected_icc_sha256=member["icc_sha256"],
            )
        )
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(f"pair dimensions differ: {row['scene_id']}")
    return arrays[0], arrays[1]


def _sample_fit_rows(
    rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    pixels_per_scene: int,
    reverse_enumeration: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sampled: list[tuple[str, np.ndarray, np.ndarray]] = []
    iterable = reversed(rows) if reverse_enumeration else rows
    for row in iterable:
        source, target = _load_pair(row, lookup)
        indexes = sample_indexes(
            row["scene_id"], source.shape[0] * source.shape[1], pixels_per_scene
        )
        sampled.append(
            (
                row["scene_id"],
                source.reshape(-1, 3)[indexes],
                target.reshape(-1, 3)[indexes],
            )
        )
    sampled.sort(key=lambda item: item[0])
    return (
        np.concatenate([item[1] for item in sampled]),
        np.concatenate([item[2] for item in sampled]),
    )


def _evaluate_row(
    row: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
    operators: dict[str, LogitAffineOperator],
) -> dict[str, Any]:
    source, target = _load_pair(row, lookup)
    identity_error = mean_oklab_error(source, target)
    outputs = {name: apply_operator(source, value) for name, value in operators.items()}
    errors = {name: mean_oklab_error(value, target) for name, value in outputs.items()}
    candidate = outputs["candidate"]
    output_delta = np.linalg.norm(
        srgb_code_to_oklab(candidate) - srgb_code_to_oklab(source), axis=-1
    )
    return {
        "scene_id": row["scene_id"],
        "directed_role_pair": f"{row['loser']}->{row['winner']}",
        "identity_error": identity_error,
        "errors": errors,
        "candidate_improvement": _relative_gain(identity_error, errors["candidate"]),
        "diagonal_relative_gain": _relative_gain(errors["diagonal"], errors["candidate"]),
        "permuted_relative_gain": _relative_gain(errors["permuted"], errors["candidate"]),
        "reverse_improvement": _relative_gain(identity_error, errors["reverse"]),
        "candidate_output_delta_e_oklab": float(np.mean(output_delta)),
        "candidate_new_exact_boundary_fraction": new_exact_boundary_fraction(
            source, candidate
        ),
        "candidate_p999_gradient_ratio": gradient_p999_ratio(source, candidate),
    }


def run(
    contract_path: Path,
    acquisition_path: Path,
    report_path: Path,
    *,
    reverse_enumeration: bool,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    acquisition_bytes = acquisition_path.read_bytes()
    contract = json.loads(contract_bytes)
    acquisition = json.loads(acquisition_bytes)
    parent = contract["parent"]
    if _sha256(acquisition_bytes) != parent["acquisition_sha256"]:
        raise ValueError("acquisition report hash drift")
    if acquisition["decision"] != parent["required_decision"]:
        raise ValueError("acquisition decision is not admitted")
    if acquisition["sealed_member_requests"] != 0:
        raise ValueError("sealed members were read before development pass")
    if acquisition["selected_identity_sha256"] != parent["selected_identity_sha256"]:
        raise ValueError("selected role identity drift")
    lookup = _lookup(acquisition)
    fit_rows = [row for row in acquisition["records"] if row["role"] == "fit"]
    calibration_rows = [
        row for row in acquisition["records"] if row["role"] == "calibration"
    ]
    fit_sources, fit_targets = _sample_fit_rows(
        fit_rows,
        lookup,
        int(contract["fit"]["pixels_per_scene"]),
        reverse_enumeration,
    )
    alpha = float(contract["fit"]["ridge_alpha"])
    candidate_raw = fit_operator_rows(fit_sources, fit_targets, ridge_alpha=alpha)
    diagonal_raw = fit_operator_rows(
        fit_sources, fit_targets, ridge_alpha=alpha, diagonal=True
    )
    target_blocks = fit_targets.reshape(len(fit_rows), -1, 3)
    permuted_targets = np.roll(target_blocks, -1, axis=0).reshape(-1, 3)
    permuted_raw = fit_operator_rows(fit_sources, permuted_targets, ridge_alpha=alpha)
    reverse_raw = fit_operator_rows(fit_targets, fit_sources, ridge_alpha=alpha)
    raw = {
        "candidate": (candidate_raw, fit_sources, fit_targets),
        "diagonal": (diagonal_raw, fit_sources, fit_targets),
        "permuted": (permuted_raw, fit_sources, permuted_targets),
        "reverse": (reverse_raw, fit_targets, fit_sources),
    }
    operators: dict[str, LogitAffineOperator] = {}
    dose_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for name, ((matrix, bias), selection_sources, selection_targets) in raw.items():
        operators[name], dose_diagnostics[name] = _select_dose(
            matrix, bias, selection_sources, selection_targets, contract
        )
    rows = [_evaluate_row(row, lookup, operators) for row in calibration_rows]
    rows.sort(key=lambda row: row["scene_id"])
    improvements = np.asarray([row["candidate_improvement"] for row in rows])
    diagonal = np.asarray([row["diagonal_relative_gain"] for row in rows])
    permuted = np.asarray([row["permuted_relative_gain"] for row in rows])
    reverse = np.asarray([row["reverse_improvement"] for row in rows])
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": float(np.mean(improvements > 0.0)),
        "candidate_improvement_median": float(np.median(improvements)),
        "candidate_improvement_worst": float(np.min(improvements)),
        "beat_diagonal_rate": float(np.mean(diagonal > 0.0)),
        "diagonal_relative_gain_median": float(np.median(diagonal)),
        "beat_label_permuted_rate": float(np.mean(permuted > 0.0)),
        "label_permuted_relative_gain_median": float(np.median(permuted)),
        "reverse_direction_improvement_rate": float(np.mean(reverse > 0.0)),
        "median_output_delta_e_oklab_from_identity": float(
            np.median([row["candidate_output_delta_e_oklab"] for row in rows])
        ),
        "maximum_new_exact_boundary_fraction": float(
            max(row["candidate_new_exact_boundary_fraction"] for row in rows)
        ),
        "maximum_p999_gradient_ratio": float(
            max(row["candidate_p999_gradient_ratio"] for row in rows)
        ),
        "directed_role_pair_counts": dict(
            sorted(Counter(row["directed_role_pair"] for row in rows).items())
        ),
    }
    gates = contract["calibration_metrics"]
    gate_results = {
        "candidate_improvement_rate": metrics["candidate_improvement_rate"]
        >= gates["candidate_improvement_rate_min"],
        "candidate_improvement_median": metrics["candidate_improvement_median"]
        >= gates["candidate_improvement_median_min"],
        "candidate_improvement_worst": metrics["candidate_improvement_worst"]
        >= gates["candidate_improvement_worst_min"],
        "beat_diagonal_rate": metrics["beat_diagonal_rate"]
        >= gates["beat_diagonal_rate_min"],
        "diagonal_relative_gain_median": metrics["diagonal_relative_gain_median"]
        >= gates["diagonal_relative_gain_median_min"],
        "beat_label_permuted_rate": metrics["beat_label_permuted_rate"]
        >= gates["beat_label_permuted_rate_min"],
        "label_permuted_relative_gain_median": metrics[
            "label_permuted_relative_gain_median"
        ]
        >= gates["label_permuted_relative_gain_median_min"],
        "reverse_direction_improvement_rate": metrics[
            "reverse_direction_improvement_rate"
        ]
        <= gates["reverse_direction_improvement_rate_max"],
        "median_output_delta_e_oklab_from_identity": metrics[
            "median_output_delta_e_oklab_from_identity"
        ]
        >= gates["median_output_delta_e_oklab_from_identity_min"],
        "new_exact_boundary_fraction": metrics["maximum_new_exact_boundary_fraction"]
        <= gates["new_exact_boundary_fraction_max"],
        "p999_gradient_ratio": metrics["maximum_p999_gradient_ratio"]
        <= gates["p999_gradient_ratio_max"],
    }
    failed = sorted(name for name, passed in gate_results.items() if not passed)
    report = {
        "schema": "neuro_film.u5_r2repid4_shared_logit_affine_d0_report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_bytes),
        "acquisition_sha256": _sha256(acquisition_bytes),
        "enumeration_normalized": True,
        "fit_sample_count": int(fit_sources.shape[0]),
        "operators": {name: _operator_payload(value) for name, value in operators.items()},
        "dose_diagnostics": dose_diagnostics,
        "calibration_rows": rows,
        "metrics": metrics,
        "gate_results": gate_results,
        "failed_gates": failed,
        "sealed_member_requests": acquisition["sealed_member_requests"],
        "automatic_pass": not failed,
        "decision": contract["decision_if_pass"] if not failed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(
        args.config,
        args.acquisition,
        args.output,
        reverse_enumeration=args.reverse,
    )
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
