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
from PIL import Image, ImageCms, features

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


def _retain_largest(retained: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    combined = (
        flattened if retained.size == 0 else np.concatenate((retained, flattened))
    )
    if combined.size <= count:
        return combined
    return combined[np.argpartition(combined, combined.size - count)[-count:]]


def _quantile_from_largest(
    retained: np.ndarray, total_count: int, quantile: float
) -> float:
    rank = (total_count - 1) * quantile
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    ordered = np.sort(retained)
    offset = total_count - ordered.size
    lower_value = ordered[lower - offset]
    upper_value = ordered[upper - offset]
    return float(lower_value + (rank - lower) * (upper_value - lower_value))


def _evaluate_arrays(
    source: np.ndarray,
    target: np.ndarray,
    operators: dict[str, LogitAffineOperator],
    *,
    row_block: int = 64,
) -> dict[str, Any]:
    height, width, _ = source.shape
    pixel_count = height * width
    gradient_count = height * (width - 1) + (height - 1) * width
    retained_count = gradient_count - int(np.floor((gradient_count - 1) * 0.999))
    identity_error_sum = 0.0
    error_sums = {name: 0.0 for name in operators}
    output_delta_sum = 0.0
    new_boundary_count = 0
    source_gradients = np.empty(0, dtype=np.float64)
    candidate_gradients = np.empty(0, dtype=np.float64)
    luma_weights = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
    for start in range(0, height, row_block):
        end = min(start + row_block, height)
        extended_end = min(end + 1, height)
        source_tile = source[start:extended_end]
        core_rows = end - start
        source_lab = srgb_code_to_oklab(source_tile[:core_rows])
        target_lab = srgb_code_to_oklab(target[start:end])
        identity_error_sum += float(
            np.sum(np.linalg.norm(source_lab - target_lab, axis=-1))
        )
        source_luma = np.asarray(source_tile, dtype=np.float64) @ luma_weights
        source_gradients = _retain_largest(
            source_gradients,
            np.abs(np.diff(source_luma[:core_rows], axis=1)),
            retained_count,
        )
        if extended_end > start + 1:
            source_gradients = _retain_largest(
                source_gradients,
                np.abs(np.diff(source_luma, axis=0)),
                retained_count,
            )
        for name, operator in operators.items():
            output_tile = apply_operator(source_tile, operator)
            output_core = output_tile[:core_rows]
            output_lab = srgb_code_to_oklab(output_core)
            error_sums[name] += float(
                np.sum(np.linalg.norm(output_lab - target_lab, axis=-1))
            )
            if name != "candidate":
                continue
            output_delta_sum += float(
                np.sum(np.linalg.norm(output_lab - source_lab, axis=-1))
            )
            source_boundary = (source_tile[:core_rows] <= 0.0) | (
                source_tile[:core_rows] >= 1.0
            )
            output_boundary = (output_core <= 0.0) | (output_core >= 1.0)
            new_boundary_count += int(
                np.count_nonzero(output_boundary & ~source_boundary)
            )
            output_luma = np.asarray(output_tile, dtype=np.float64) @ luma_weights
            candidate_gradients = _retain_largest(
                candidate_gradients,
                np.abs(np.diff(output_luma[:core_rows], axis=1)),
                retained_count,
            )
            if extended_end > start + 1:
                candidate_gradients = _retain_largest(
                    candidate_gradients,
                    np.abs(np.diff(output_luma, axis=0)),
                    retained_count,
                )
    source_p999 = _quantile_from_largest(source_gradients, gradient_count, 0.999)
    candidate_p999 = _quantile_from_largest(candidate_gradients, gradient_count, 0.999)
    return {
        "identity_error": identity_error_sum / pixel_count,
        "errors": {name: value / pixel_count for name, value in error_sums.items()},
        "candidate_output_delta_e_oklab": output_delta_sum / pixel_count,
        "candidate_new_exact_boundary_fraction": new_boundary_count / source.size,
        "candidate_p999_gradient_ratio": candidate_p999 / max(source_p999, 1.0e-12),
    }


def _evaluate_row(
    row: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
    operators: dict[str, LogitAffineOperator],
) -> dict[str, Any]:
    source, target = _load_pair(row, lookup)
    evaluated = _evaluate_arrays(source, target, operators)
    identity_error = evaluated["identity_error"]
    errors = evaluated["errors"]
    return {
        "scene_id": row["scene_id"],
        "directed_role_pair": f"{row['loser']}->{row['winner']}",
        "identity_error": identity_error,
        "errors": errors,
        "candidate_improvement": _relative_gain(identity_error, errors["candidate"]),
        "diagonal_relative_gain": _relative_gain(
            errors["diagonal"], errors["candidate"]
        ),
        "permuted_relative_gain": _relative_gain(
            errors["permuted"], errors["candidate"]
        ),
        "reverse_improvement": _relative_gain(identity_error, errors["reverse"]),
        "candidate_output_delta_e_oklab": evaluated["candidate_output_delta_e_oklab"],
        "candidate_new_exact_boundary_fraction": evaluated[
            "candidate_new_exact_boundary_fraction"
        ],
        "candidate_p999_gradient_ratio": evaluated["candidate_p999_gradient_ratio"],
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
    implementation = {
        name: _sha256((ROOT / binding["path"]).read_bytes())
        for name, binding in contract["implementation"].items()
    }
    if any(
        implementation[name] != binding["sha256"]
        for name, binding in contract["implementation"].items()
    ):
        raise ValueError("implementation identity drift")
    parent = contract["parent"]
    if _sha256(acquisition_bytes) != parent["acquisition_sha256"]:
        raise ValueError("acquisition report hash drift")
    if acquisition["decision"] != parent["required_decision"]:
        raise ValueError("acquisition decision is not admitted")
    if acquisition["sealed_member_requests"] != 0:
        raise ValueError("sealed members were read before development pass")
    if acquisition["selected_identity_sha256"] != parent["selected_identity_sha256"]:
        raise ValueError("selected role identity drift")
    ingress = contract["ingress"]
    observed_icc = {
        member["icc_sha256"]
        for record in acquisition["records"]
        for member in record["members"]
    }
    if observed_icc != {ingress["required_icc_sha256"]}:
        raise ValueError("ICC population identity drift")
    observed_runtime = {
        "pillow": Image.__version__,
        "littlecms": ImageCms.core.littlecms_version,
        "libjpeg": features.version("jpg"),
    }
    if observed_runtime != ingress["runtime_versions"]:
        raise ValueError("ICC conversion runtime drift")
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
        "implementation_sha256": implementation,
        "ingress": {"icc_sha256": next(iter(observed_icc)), **observed_runtime},
        "enumeration_normalized": True,
        "fit_sample_count": int(fit_sources.shape[0]),
        "operators": {
            name: _operator_payload(value) for name, value in operators.items()
        },
        "dose_diagnostics": dose_diagnostics,
        "calibration_rows": rows,
        "metrics": metrics,
        "gate_results": gate_results,
        "failed_gates": failed,
        "sealed_member_requests": acquisition["sealed_member_requests"],
        "automatic_pass": not failed,
        "decision": contract["decision_if_pass"]
        if not failed
        else contract["decision_if_fail"],
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
