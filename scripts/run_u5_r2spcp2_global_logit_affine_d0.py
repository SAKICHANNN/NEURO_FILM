#!/usr/bin/env python3
"""Fit and score the frozen SPCP2 shared global logit-affine operator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    ROOT,
    _canonical_bytes,
    _sha256,
    _stable_id,
)
from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    dose_operator,
    fit_operator_rows,
    gradient_p999_ratio,
    matrix_diagnostics,
    mean_oklab_error,
    new_exact_boundary_fraction,
    sample_indexes,
    srgb_code_to_oklab,
)

DEFAULT_CONTRACT = ROOT / "configs/u5_r2spcp2_global_logit_affine_preference_d0_v1.json"
DEFAULT_MANIFEST = ROOT / "manifests/u5_r2spcp2_global_logit_affine_roles_v1.json"
DEFAULT_ACQUISITION = ROOT / "outputs/eval/u5_r2spcp2_global_logit_affine_v1/acquisition_a.json"
DEFAULT_REPORT = ROOT / "outputs/eval/u5_r2spcp2_global_logit_affine_v1/formal_result.json"


def _load_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(f"input hash drift: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"decode mode drift: {path}")
        if image.width < 256 or image.height < 256:
            raise ValueError(f"image geometry below minimum: {path}")
        if image.info.get("icc_profile"):
            raise ValueError(f"embedded ICC is not admitted: {path}")
        values = np.asarray(image, dtype=np.uint8)
    return np.asarray(values, dtype=np.float32) / np.float32(255.0)


def _acquisition_lookup(acquisition: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (record["scene_id"], record["side"]): record for record in acquisition["records"]
    }


def _load_pair(
    row: dict[str, Any], lookup: dict[tuple[str, str], dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for side in ("loser", "winner"):
        record = lookup[(row["scene_id"], side)]
        arrays.append(_load_rgb(ROOT / record["relative_path"], record["sha256"]))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(f"pair dimensions differ: {row['scene_id']}")
    return arrays[0], arrays[1]


def _sample_fit_rows(
    rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    pixels_per_scene: int,
    reverse_enumeration: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    iterable = list(reversed(rows)) if reverse_enumeration else rows
    for row in iterable:
        source, target = _load_pair(row, lookup)
        indexes = sample_indexes(
            row["scene_id"], source.shape[0] * source.shape[1], pixels_per_scene
        )
        source_samples.append(source.reshape(-1, 3)[indexes])
        target_samples.append(target.reshape(-1, 3)[indexes])
    # Canonicalize accumulation order independently of enumeration order.
    if reverse_enumeration:
        source_samples.reverse()
        target_samples.reverse()
    return np.concatenate(source_samples), np.concatenate(target_samples)


def _select_dose(
    matrix: np.ndarray,
    bias: np.ndarray,
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    contract: dict[str, Any],
) -> tuple[LogitAffineOperator, list[dict[str, Any]]]:
    target = np.asarray(target_rows, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    gates = contract["fit"]["matrix_gates"]
    for dose in contract["fit"]["dose_grid"]:
        operator = dose_operator(matrix, bias, float(dose))
        matrix_metrics = matrix_diagnostics(operator)
        valid = (
            matrix_metrics["determinant"] >= gates["determinant_min"]
            and matrix_metrics["condition_number"] <= gates["condition_number_max"]
            and matrix_metrics["minimum_singular_value"] >= gates["minimum_singular_value"]
        )
        predicted = apply_operator(source_rows, operator)
        diagnostics.append(
            {
                "dose": float(dose),
                **matrix_metrics,
                "matrix_gate_pass": bool(valid),
                "fit_code_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
            }
        )
    eligible = [row for row in diagnostics if row["matrix_gate_pass"]]
    if not eligible:
        raise ValueError("no dose passes the frozen analytic matrix gates")
    selected = min(eligible, key=lambda row: (row["fit_code_rmse"], row["dose"]))
    return dose_operator(matrix, bias, selected["dose"]), diagnostics


def _operator_payload(operator: LogitAffineOperator) -> dict[str, Any]:
    return {
        "matrix": operator.matrix.tolist(),
        "bias": operator.bias.tolist(),
        "dose": operator.dose,
        **matrix_diagnostics(operator),
    }


def _relative_gain(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(baseline, 1.0e-12))


def _evaluate_row(
    row: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
    operators: dict[str, LogitAffineOperator],
) -> dict[str, Any]:
    source, target = _load_pair(row, lookup)
    identity_error = mean_oklab_error(source, target)
    outputs = {name: apply_operator(source, operator) for name, operator in operators.items()}
    errors = {name: mean_oklab_error(output, target) for name, output in outputs.items()}
    candidate = outputs["candidate"]
    output_delta = np.linalg.norm(
        srgb_code_to_oklab(candidate) - srgb_code_to_oklab(source), axis=-1
    )
    return {
        "scene_id": row["scene_id"],
        "identity_error": identity_error,
        "errors": errors,
        "candidate_improvement": _relative_gain(identity_error, errors["candidate"]),
        "diagonal_relative_gain": _relative_gain(errors["diagonal"], errors["candidate"]),
        "permuted_relative_gain": _relative_gain(errors["permuted"], errors["candidate"]),
        "reverse_improvement": _relative_gain(identity_error, errors["reverse"]),
        "candidate_output_delta_e_oklab": float(np.mean(output_delta)),
        "candidate_new_exact_boundary_fraction": new_exact_boundary_fraction(source, candidate),
        "candidate_p999_gradient_ratio": gradient_p999_ratio(source, candidate),
    }


def run(
    contract_path: Path,
    manifest_path: Path,
    acquisition_path: Path,
    report_path: Path,
    reverse_enumeration: bool,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    acquisition_bytes = acquisition_path.read_bytes()
    contract = json.loads(contract_bytes)
    manifest = json.loads(manifest_bytes)
    acquisition = json.loads(acquisition_bytes)
    if acquisition["status"] != "PASS_FIT_CAL_SELECTED_PAIR_ACQUISITION":
        raise ValueError("acquisition is not admitted")
    if acquisition["contract_sha256"] != _sha256(contract_bytes):
        raise ValueError("contract/acquisition binding drift")
    if acquisition["roles_manifest_sha256"] != _sha256(manifest_bytes):
        raise ValueError("manifest/acquisition binding drift")
    if acquisition["sealed_payload_count"] != 0:
        raise ValueError("sealed pixels were read before development pass")
    lookup = _acquisition_lookup(acquisition)
    fit_rows = [row for row in manifest["selected_rows"] if row["role"] == "fit"]
    calibration_rows = [
        row for row in manifest["selected_rows"] if row["role"] == "calibration"
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
    permuted_targets = fit_targets.reshape(len(fit_rows), -1, 3)
    permuted_targets = np.roll(permuted_targets, -1, axis=0).reshape(-1, 3)
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
    iterable = list(reversed(calibration_rows)) if reverse_enumeration else calibration_rows
    rows = [_evaluate_row(row, lookup, operators) for row in iterable]
    rows.sort(key=lambda row: row["scene_id"])
    improvements = np.asarray([row["candidate_improvement"] for row in rows])
    diagonal_gains = np.asarray([row["diagonal_relative_gain"] for row in rows])
    permuted_gains = np.asarray([row["permuted_relative_gain"] for row in rows])
    reverse_improvements = np.asarray([row["reverse_improvement"] for row in rows])
    gates = contract["calibration_metrics"]
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": float(np.mean(improvements > 0.0)),
        "candidate_improvement_median": float(np.median(improvements)),
        "candidate_improvement_worst": float(np.min(improvements)),
        "beat_diagonal_rate": float(np.mean(diagonal_gains > 0.0)),
        "diagonal_relative_gain_median": float(np.median(diagonal_gains)),
        "beat_label_permuted_rate": float(np.mean(permuted_gains > 0.0)),
        "label_permuted_relative_gain_median": float(np.median(permuted_gains)),
        "reverse_direction_improvement_rate": float(np.mean(reverse_improvements > 0.0)),
        "median_output_delta_e_oklab_from_identity": float(
            np.median([row["candidate_output_delta_e_oklab"] for row in rows])
        ),
        "maximum_new_exact_boundary_fraction": float(
            max(row["candidate_new_exact_boundary_fraction"] for row in rows)
        ),
        "maximum_p999_gradient_ratio": float(
            max(row["candidate_p999_gradient_ratio"] for row in rows)
        ),
    }
    gate_results = {
        "candidate_improvement_rate": metrics["candidate_improvement_rate"]
        >= gates["candidate_improvement_rate_min"],
        "candidate_improvement_median": metrics["candidate_improvement_median"]
        >= gates["candidate_improvement_median_min"],
        "candidate_improvement_worst": metrics["candidate_improvement_worst"]
        >= gates["candidate_improvement_worst_min"],
        "beat_diagonal_rate": metrics["beat_diagonal_rate"] >= gates["beat_diagonal_rate_min"],
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
        "schema": "neuro-film.u5-r2spcp2-global-logit-affine-report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_bytes),
        "roles_manifest_sha256": _sha256(manifest_bytes),
        "acquisition_sha256": _sha256(acquisition_bytes),
        "enumeration_normalized": True,
        "fit_sample_count": int(fit_sources.shape[0]),
        "operators": {name: _operator_payload(operator) for name, operator in operators.items()},
        "dose_diagnostics": dose_diagnostics,
        "calibration_rows": rows,
        "metrics": metrics,
        "gate_results": gate_results,
        "failed_gates": failed,
        "sealed_payload_count": acquisition["sealed_payload_count"],
        "status": "PASS_DEVELOPMENT_GATES" if not failed else "FAIL_CLOSED_DEVELOPMENT_GATES",
        "decision": (
            "open_exact_sealed_confirmation"
            if not failed
            else "close_exact_global_logit_affine_population_prior"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_result_id"] = _stable_id(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(
        args.contract.resolve(),
        args.manifest.resolve(),
        args.acquisition.resolve(),
        args.report.resolve(),
        args.reverse,
    )
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
