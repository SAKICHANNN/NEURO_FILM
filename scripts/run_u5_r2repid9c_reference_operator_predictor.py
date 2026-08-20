#!/usr/bin/env python3
"""Run the frozen REPID reference-only bounded-operator predictor D0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.repid_reference_operator_predictor import (
    decode_safe_effect,
    evaluate_effects,
    fit_ridge,
    fit_scene_effect,
    load_jpeg_as_srgb,
    load_original_srgb,
    matrix_diagnostics,
    model_payload,
    predict_ridge,
    reference_descriptor,
    relative_gain,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _winner_lookup(acquisition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["scene_id"]: next(
            member for member in row["members"] if member["endpoint"] == "winner"
        )
        for row in acquisition["records"]
    }


def _load_winner(member: dict[str, Any]) -> np.ndarray:
    return load_jpeg_as_srgb(
        ROOT / member["relative_path"],
        expected_sha256=member["sha256"],
        expected_icc_sha256=member["icc_sha256"],
    )


def _summarize(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    improvements = np.asarray([row["candidate_improvement"] for row in rows])
    global_gains = np.asarray([row["gain_vs_global"] for row in rows])
    permuted_gains = np.asarray([row["gain_vs_permuted"] for row in rows])
    gates_spec = config["calibration_gates"]
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": float(np.mean(improvements > 0.0)),
        "candidate_improvement_median": float(np.median(improvements)),
        "candidate_improvement_worst": float(np.min(improvements)),
        "beat_global_rate": float(np.mean(global_gains > 0.0)),
        "gain_vs_global_median": float(np.median(global_gains)),
        "beat_permuted_rate": float(np.mean(permuted_gains > 0.0)),
        "gain_vs_permuted_median": float(np.median(permuted_gains)),
        "median_output_delta_e_oklab": float(
            np.median([row["candidate_output_delta_e_oklab"] for row in rows])
        ),
        "maximum_new_exact_boundary_fraction": float(
            max(row["candidate_new_exact_boundary_fraction"] for row in rows)
        ),
        "maximum_p999_gradient_ratio": float(
            max(row["candidate_p999_gradient_ratio"] for row in rows)
        ),
        "minimum_determinant": float(
            min(row["candidate_matrix"]["determinant"] for row in rows)
        ),
        "maximum_condition_number": float(
            max(row["candidate_matrix"]["condition_number"] for row in rows)
        ),
        "minimum_singular_value": float(
            min(row["candidate_matrix"]["minimum_singular_value"] for row in rows)
        ),
        "median_oracle_improvement": float(
            np.median([row["oracle_improvement"] for row in rows])
        ),
    }
    gates = {
        "candidate_improvement_rate": metrics["candidate_improvement_rate"]
        >= gates_spec["candidate_improvement_rate_min"],
        "candidate_improvement_median": metrics["candidate_improvement_median"]
        >= gates_spec["candidate_improvement_median_min"],
        "candidate_improvement_worst": metrics["candidate_improvement_worst"]
        >= gates_spec["candidate_improvement_worst_min"],
        "beat_global_rate": metrics["beat_global_rate"]
        >= gates_spec["beat_global_rate_min"],
        "gain_vs_global_median": metrics["gain_vs_global_median"]
        >= gates_spec["gain_vs_global_median_min"],
        "beat_permuted_rate": metrics["beat_permuted_rate"]
        >= gates_spec["beat_permuted_rate_min"],
        "gain_vs_permuted_median": metrics["gain_vs_permuted_median"]
        >= gates_spec["gain_vs_permuted_median_min"],
        "style_delta": metrics["median_output_delta_e_oklab"]
        >= gates_spec["median_output_delta_e_oklab_min"],
        "new_boundary": metrics["maximum_new_exact_boundary_fraction"]
        <= gates_spec["new_exact_boundary_fraction_max"],
        "gradient": metrics["maximum_p999_gradient_ratio"]
        <= gates_spec["p999_gradient_ratio_max"],
    }
    return metrics, gates


def run(
    config_path: Path,
    original_root: Path,
    original_report_path: Path,
    report_path: Path,
    *,
    reverse: bool,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    for binding in config["implementation"].values():
        if _sha256((ROOT / binding["path"]).read_bytes()) != binding["sha256"]:
            raise ValueError("implementation identity drift")
    rendered_bytes = (
        ROOT / config["parents"]["rendered_acquisition_path"]
    ).read_bytes()
    if _sha256(rendered_bytes) != config["parents"]["rendered_acquisition_sha256"]:
        raise ValueError("rendered acquisition drift")
    original_evidence_bytes = (
        ROOT / config["parents"]["original_acquisition_evidence_path"]
    ).read_bytes()
    if (
        _sha256(original_evidence_bytes)
        != config["parents"]["original_acquisition_evidence_sha256"]
    ):
        raise ValueError("original acquisition evidence drift")
    if (
        json.loads(original_evidence_bytes)["decision"]
        != "open_repid_reference_only_bounded_parameter_predictor_d0"
    ):
        raise ValueError("original acquisition is not admitted")
    original_report_bytes = original_report_path.read_bytes()
    rendered = json.loads(rendered_bytes)
    original_report = json.loads(original_report_bytes)
    if (
        _sha256(original_report_bytes)
        != config["parents"]["original_acquisition_sha256"]
    ):
        raise ValueError("original acquisition drift")
    originals = {row["scene_id"]: row for row in original_report["records"]}
    winners = _winner_lookup(rendered)
    ordered = sorted(
        rendered["records"], key=lambda row: row["scene_id"], reverse=reverse
    )
    fit_rows = [row for row in ordered if row["role"] == "fit"]
    calibration_rows = [row for row in ordered if row["role"] == "calibration"]
    if len(fit_rows) != int(config["roles"]["fit_scene_count"]) or len(
        calibration_rows
    ) != int(config["roles"]["calibration_scene_count"]):
        raise ValueError("fit/calibration role drift")
    if set(originals) != {row["scene_id"] for row in ordered}:
        raise ValueError("original/render scene identity drift")
    fit_features: list[np.ndarray] = []
    fit_parameters: list[np.ndarray] = []
    fit_labels: list[dict[str, Any]] = []
    for row in fit_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        original_record = originals[scene_id]
        original = load_original_srgb(
            original_root / original_record["logical_path"],
            expected_sha256=original_record["sha256"],
        )
        parameters, diagnostics = fit_scene_effect(
            original, winner, scene_id, config["operator"]
        )
        fit_features.append(reference_descriptor(winner, config["descriptor"]))
        fit_parameters.append(parameters)
        fit_labels.append(
            {
                "scene_id": scene_id,
                "parameters": parameters.tolist(),
                "diagnostics": diagnostics,
            }
        )
    canonical_fit = sorted(
        zip(
            [row["scene_id"] for row in fit_rows],
            fit_features,
            fit_parameters,
            strict=True,
        ),
        key=lambda item: item[0],
    )
    features = np.stack([item[1] for item in canonical_fit])
    targets = np.stack([item[2] for item in canonical_fit])
    model = fit_ridge(
        features, targets, alpha=float(config["predictor"]["ridge_alpha"])
    )
    permuted = np.roll(
        targets, int(config["controls"]["cyclic_parameter_shift"]), axis=0
    )
    permuted_model = fit_ridge(
        features, permuted, alpha=float(config["predictor"]["ridge_alpha"])
    )
    global_parameters = np.mean(targets, axis=0)
    calibration_references: dict[str, np.ndarray] = {}
    calibration_features: dict[str, np.ndarray] = {}
    for row in calibration_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        calibration_references[scene_id] = winner
        calibration_features[scene_id] = reference_descriptor(
            winner, config["descriptor"]
        )
    canonical_cal_ids = sorted(calibration_features)
    cal_matrix = np.stack(
        [calibration_features[scene_id] for scene_id in canonical_cal_ids]
    )
    predictions = predict_ridge(model, cal_matrix)
    permuted_predictions = predict_ridge(permuted_model, cal_matrix)
    freeze_payload = {
        "scene_ids": canonical_cal_ids,
        "predictions": predictions.tolist(),
        "permuted_predictions": permuted_predictions.tolist(),
        "global_parameters": global_parameters.tolist(),
    }
    prediction_freeze_sha256 = _canonical_sha(freeze_payload)
    rows = []
    for index, scene_id in enumerate(canonical_cal_ids):
        original_record = originals[scene_id]
        original = load_original_srgb(
            original_root / original_record["logical_path"],
            expected_sha256=original_record["sha256"],
        )
        winner = calibration_references[scene_id]
        candidate, candidate_clips = decode_safe_effect(
            predictions[index], config["operator"]
        )
        global_operator, global_clips = decode_safe_effect(
            global_parameters, config["operator"]
        )
        permuted_operator, permuted_clips = decode_safe_effect(
            permuted_predictions[index], config["operator"]
        )
        oracle_parameters, _ = fit_scene_effect(
            original, winner, scene_id, config["operator"]
        )
        oracle, _ = decode_safe_effect(oracle_parameters, config["operator"])
        evaluated = evaluate_effects(
            original,
            winner,
            {
                "candidate": candidate,
                "global": global_operator,
                "permuted": permuted_operator,
                "oracle": oracle,
            },
        )
        errors = evaluated["errors"]
        identity = evaluated["identity_error"]
        rows.append(
            {
                "scene_id": scene_id,
                "candidate_improvement": relative_gain(identity, errors["candidate"]),
                "gain_vs_global": relative_gain(errors["global"], errors["candidate"]),
                "gain_vs_permuted": relative_gain(
                    errors["permuted"], errors["candidate"]
                ),
                "oracle_improvement": relative_gain(identity, errors["oracle"]),
                "identity_error": identity,
                "errors": errors,
                "candidate_matrix": matrix_diagnostics(candidate),
                "candidate_dose": candidate.dose,
                "parameter_clip_counts": {
                    "candidate": candidate_clips,
                    "global": global_clips,
                    "permuted": permuted_clips,
                },
                "candidate_output_delta_e_oklab": evaluated[
                    "candidate_output_delta_e_oklab"
                ],
                "candidate_new_exact_boundary_fraction": evaluated[
                    "candidate_new_exact_boundary_fraction"
                ],
                "candidate_p999_gradient_ratio": evaluated[
                    "candidate_p999_gradient_ratio"
                ],
            }
        )
    metrics, gates = _summarize(rows, config)
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_bytes),
        "fit_scene_count": len(fit_rows),
        "calibration_scene_count": len(calibration_rows),
        "calibration_original_reads_before_prediction_freeze": 0,
        "sealed_requests": 0,
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "model": model_payload(model),
        "permuted_model": model_payload(permuted_model),
        "fit_label_identity_sha256": _canonical_sha(
            sorted(fit_labels, key=lambda row: row["scene_id"])
        ),
        "rows": rows,
        "metrics": metrics,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(encoded)
    return {"report": report, "sha256": _sha256(encoded)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        args.original_root.resolve(),
        args.original_report.resolve(),
        args.report.resolve(),
        reverse=args.reverse,
    )
    print(
        json.dumps(
            {"decision": result["report"]["decision"], "sha256": result["sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
