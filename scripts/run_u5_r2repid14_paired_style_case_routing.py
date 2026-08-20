#!/usr/bin/env python3
"""Run the frozen REPID paired-style transparent case-routing D0."""

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

from src.eval.repid_paired_style_latent import spatial_descriptor
from src.eval.repid_paired_style_retrieval import (
    predict_case_routes,
    predict_latent_oracle,
    train_case_bank,
)
from src.eval.repid_reference_operator_predictor import (
    decode_safe_effect,
    evaluate_effects,
    fit_scene_effect,
    load_jpeg_as_srgb,
    load_original_srgb,
    matrix_diagnostics,
    relative_gain,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _winner_lookup(acquisition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["scene_id"]: next(member for member in row["members"] if member["endpoint"] == "winner")
        for row in acquisition["records"]
    }


def _load_winner(member: dict[str, Any]) -> np.ndarray:
    return load_jpeg_as_srgb(
        ROOT / member["relative_path"],
        expected_sha256=member["sha256"],
        expected_icc_sha256=member["icc_sha256"],
    )


def _rate(values: np.ndarray) -> float:
    return float(np.mean(values > 0.0))


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any], operator: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    comparisons = {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in (
            "candidate_improvement",
            "gain_vs_global",
            "gain_vs_top1",
            "gain_vs_medoid",
            "gain_vs_repid13_student",
            "gain_vs_cyclic",
            "paired_latent_oracle_improvement",
        )
    }
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": _rate(comparisons["candidate_improvement"]),
        "candidate_improvement_median": float(np.median(comparisons["candidate_improvement"])),
        "candidate_improvement_worst": float(np.min(comparisons["candidate_improvement"])),
        "beat_global_rate": _rate(comparisons["gain_vs_global"]),
        "gain_vs_global_median": float(np.median(comparisons["gain_vs_global"])),
        "beat_top1_rate": _rate(comparisons["gain_vs_top1"]),
        "gain_vs_top1_median": float(np.median(comparisons["gain_vs_top1"])),
        "beat_medoid_rate": _rate(comparisons["gain_vs_medoid"]),
        "gain_vs_medoid_median": float(np.median(comparisons["gain_vs_medoid"])),
        "beat_repid13_student_rate": _rate(comparisons["gain_vs_repid13_student"]),
        "gain_vs_repid13_student_median": float(np.median(comparisons["gain_vs_repid13_student"])),
        "beat_cyclic_rate": _rate(comparisons["gain_vs_cyclic"]),
        "gain_vs_cyclic_median": float(np.median(comparisons["gain_vs_cyclic"])),
        "paired_latent_oracle_improvement_rate": _rate(comparisons["paired_latent_oracle_improvement"]),
        "paired_latent_oracle_improvement_median": float(np.median(comparisons["paired_latent_oracle_improvement"])),
        "median_output_delta_e_oklab": float(np.median([row["candidate_output_delta_e_oklab"] for row in rows])),
        "maximum_new_exact_boundary_fraction": float(max(row["candidate_new_exact_boundary_fraction"] for row in rows)),
        "maximum_p999_gradient_ratio": float(max(row["candidate_p999_gradient_ratio"] for row in rows)),
        "minimum_determinant": float(min(row["candidate_matrix"]["determinant"] for row in rows)),
        "maximum_condition_number": float(max(row["candidate_matrix"]["condition_number"] for row in rows)),
        "minimum_singular_value": float(min(row["candidate_matrix"]["minimum_singular_value"] for row in rows)),
    }
    spec = config["calibration_gates"]
    matrix_spec = operator["matrix_gates"]
    gates = {
        key: bool(value)
        for key, value in {
            "candidate_improvement_rate": metrics["candidate_improvement_rate"] >= spec["candidate_improvement_rate_min"],
            "candidate_improvement_median": metrics["candidate_improvement_median"] >= spec["candidate_improvement_median_min"],
            "candidate_improvement_worst": metrics["candidate_improvement_worst"] >= spec["candidate_improvement_worst_min"],
            "beat_global_rate": metrics["beat_global_rate"] >= spec["beat_global_rate_min"],
            "gain_vs_global_median": metrics["gain_vs_global_median"] >= spec["gain_vs_global_median_min"],
            "beat_top1_rate": metrics["beat_top1_rate"] >= spec["beat_top1_rate_min"],
            "gain_vs_top1_median": metrics["gain_vs_top1_median"] >= spec["gain_vs_top1_median_min"],
            "beat_medoid_rate": metrics["beat_medoid_rate"] >= spec["beat_medoid_rate_min"],
            "gain_vs_medoid_median": metrics["gain_vs_medoid_median"] >= spec["gain_vs_medoid_median_min"],
            "beat_repid13_student_rate": metrics["beat_repid13_student_rate"] >= spec["beat_repid13_student_rate_min"],
            "gain_vs_repid13_student_median": metrics["gain_vs_repid13_student_median"] >= spec["gain_vs_repid13_student_median_min"],
            "beat_cyclic_rate": metrics["beat_cyclic_rate"] >= spec["beat_cyclic_rate_min"],
            "gain_vs_cyclic_median": metrics["gain_vs_cyclic_median"] >= spec["gain_vs_cyclic_median_min"],
            "paired_latent_oracle_improvement_rate": metrics["paired_latent_oracle_improvement_rate"] >= spec["paired_latent_oracle_improvement_rate_min"],
            "paired_latent_oracle_improvement_median": metrics["paired_latent_oracle_improvement_median"] >= spec["paired_latent_oracle_improvement_median_min"],
            "style_delta": metrics["median_output_delta_e_oklab"] >= spec["median_output_delta_e_oklab_min"],
            "new_boundary": metrics["maximum_new_exact_boundary_fraction"] <= spec["new_exact_boundary_fraction_max"],
            "gradient": metrics["maximum_p999_gradient_ratio"] <= spec["p999_gradient_ratio_max"],
            "matrix_safety": metrics["minimum_determinant"] >= matrix_spec["determinant_min"] and metrics["maximum_condition_number"] <= matrix_spec["condition_number_max"] and metrics["minimum_singular_value"] >= matrix_spec["minimum_singular_value"],
        }.items()
    }
    if not all(np.isfinite(list(metrics.values()))):
        raise ValueError("non-finite aggregate metric")
    return metrics, gates


def run(config_path: Path, original_root: Path, original_report_path: Path, report_path: Path, *, reverse: bool) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    parent = config["parents"]
    repid13_config = json.loads((ROOT / parent["repid13_config_path"]).read_bytes())
    repid13_config_bytes = (ROOT / parent["repid13_config_path"]).read_bytes()
    evidence_bytes = (ROOT / parent["repid13_evidence_path"]).read_bytes()
    formal_bytes = (ROOT / parent["repid13_formal_report_path"]).read_bytes()
    if _sha256(evidence_bytes) != parent["repid13_evidence_sha256"] or json.loads(evidence_bytes)["decision"] != parent["repid13_required_decision"]:
        raise ValueError("REPID13 evidence drift")
    if _sha256(formal_bytes) != parent["repid13_formal_report_sha256"]:
        raise ValueError("REPID13 formal report drift")
    formal = json.loads(formal_bytes)
    if formal["config_sha256"] != _sha256(repid13_config_bytes):
        raise ValueError("REPID13 config/formal binding drift")
    if formal["decision"] != parent["repid13_required_decision"]:
        raise ValueError("REPID13 formal decision drift")

    rendered_bytes = (ROOT / repid13_config["parents"]["rendered_acquisition_path"]).read_bytes()
    original_evidence_bytes = (ROOT / repid13_config["parents"]["original_acquisition_evidence_path"]).read_bytes()
    original_report_bytes = original_report_path.read_bytes()
    if _sha256(rendered_bytes) != repid13_config["parents"]["rendered_acquisition_sha256"]:
        raise ValueError("rendered acquisition drift")
    if _sha256(original_evidence_bytes) != repid13_config["parents"]["original_acquisition_evidence_sha256"]:
        raise ValueError("original acquisition evidence drift")
    if _sha256(original_report_bytes) != repid13_config["parents"]["original_acquisition_sha256"]:
        raise ValueError("original acquisition drift")

    rendered = json.loads(rendered_bytes)
    original_report = json.loads(original_report_bytes)
    originals = {row["scene_id"]: row for row in original_report["records"]}
    winners = _winner_lookup(rendered)
    ordered = sorted(rendered["records"], key=lambda row: row["scene_id"], reverse=reverse)
    fit_rows = [row for row in ordered if row["role"] == "fit"]
    calibration_rows = [row for row in ordered if row["role"] == "calibration"]
    if len(fit_rows) != config["roles"]["fit_scene_count"] or len(calibration_rows) != config["roles"]["calibration_scene_count"]:
        raise ValueError("fit/calibration role drift")
    if set(originals) != {row["scene_id"] for row in ordered}:
        raise ValueError("original/render scene identity drift")

    descriptor = repid13_config["descriptor"]
    operator = repid13_config["operator"]
    fit_payload = []
    for row in fit_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        record = originals[scene_id]
        original = load_original_srgb(original_root / record["logical_path"], expected_sha256=record["sha256"])
        parameters, diagnostics = fit_scene_effect(original, winner, scene_id, operator)
        fit_payload.append((scene_id, spatial_descriptor(winner, descriptor), spatial_descriptor(original, descriptor), parameters, diagnostics))
    fit_payload.sort(key=lambda item: item[0])
    fit_ids = [item[0] for item in fit_payload]
    after_features = np.stack([item[1] for item in fit_payload])
    paired_deltas = np.stack([item[1] - item[2] for item in fit_payload])
    fit_parameters = np.stack([item[3] for item in fit_payload])
    model = train_case_bank(
        after_features,
        paired_deltas,
        fit_parameters,
        latent_spec=repid13_config["latent"],
        retrieval_spec=config["retrieval"],
    )

    calibration_references: dict[str, np.ndarray] = {}
    calibration_features: dict[str, np.ndarray] = {}
    for row in calibration_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        calibration_references[scene_id] = winner
        calibration_features[scene_id] = spatial_descriptor(winner, descriptor)
    calibration_ids = sorted(calibration_features)
    feature_matrix = np.stack([calibration_features[scene_id] for scene_id in calibration_ids])
    predictions = predict_case_routes(model, feature_matrix, config["retrieval"])
    prediction_freeze = {
        "scene_ids": calibration_ids,
        "candidate_parameters": predictions["candidate_parameters"].tolist(),
        "top1_parameters": predictions["top1_parameters"].tolist(),
        "medoid_parameters": predictions["medoid_parameters"].tolist(),
        "global_parameters": predictions["global_parameters"].tolist(),
        "repid13_student_parameters": predictions["repid13_student_parameters"].tolist(),
        "cyclic_parameters": predictions["cyclic_parameters"].tolist(),
        "neighbor_scene_ids": [[fit_ids[index] for index in row] for row in predictions["neighbor_indices"]],
        "neighbor_weights": predictions["neighbor_weights"],
    }
    prediction_freeze_sha256 = _canonical_sha(prediction_freeze)

    calibration_originals: dict[str, np.ndarray] = {}
    actual_deltas = []
    for scene_id in calibration_ids:
        record = originals[scene_id]
        original = load_original_srgb(original_root / record["logical_path"], expected_sha256=record["sha256"])
        calibration_originals[scene_id] = original
        actual_deltas.append(calibration_features[scene_id] - spatial_descriptor(original, descriptor))
    oracle = predict_latent_oracle(model, np.stack(actual_deltas), config["retrieval"])

    rows: list[dict[str, Any]] = []
    parameter_names = (
        "candidate",
        "global",
        "top1",
        "medoid",
        "repid13_student",
        "cyclic",
        "paired_latent_oracle",
    )
    for index, scene_id in enumerate(calibration_ids):
        encoded_parameters = {
            "candidate": predictions["candidate_parameters"][index],
            "global": predictions["global_parameters"][index],
            "top1": predictions["top1_parameters"][index],
            "medoid": predictions["medoid_parameters"][index],
            "repid13_student": predictions["repid13_student_parameters"][index],
            "cyclic": predictions["cyclic_parameters"][index],
            "paired_latent_oracle": oracle["parameters"][index],
        }
        decoded = {}
        clip_counts = {}
        for name in parameter_names:
            decoded[name], clip_counts[name] = decode_safe_effect(encoded_parameters[name], operator)
        evaluated = evaluate_effects(calibration_originals[scene_id], calibration_references[scene_id], decoded)
        errors = evaluated["errors"]
        identity = evaluated["identity_error"]
        rows.append({
            "scene_id": scene_id,
            "candidate_improvement": relative_gain(identity, errors["candidate"]),
            "gain_vs_global": relative_gain(errors["global"], errors["candidate"]),
            "gain_vs_top1": relative_gain(errors["top1"], errors["candidate"]),
            "gain_vs_medoid": relative_gain(errors["medoid"], errors["candidate"]),
            "gain_vs_repid13_student": relative_gain(errors["repid13_student"], errors["candidate"]),
            "gain_vs_cyclic": relative_gain(errors["cyclic"], errors["candidate"]),
            "paired_latent_oracle_improvement": relative_gain(identity, errors["paired_latent_oracle"]),
            "identity_error": identity,
            "errors": errors,
            "candidate_matrix": matrix_diagnostics(decoded["candidate"]),
            "candidate_dose": decoded["candidate"].dose,
            "parameter_clip_counts": clip_counts,
            "candidate_neighbor_scene_ids": prediction_freeze["neighbor_scene_ids"][index],
            "candidate_neighbor_weights": prediction_freeze["neighbor_weights"][index],
            "oracle_neighbor_scene_ids": [fit_ids[item] for item in oracle["neighbor_indices"][index]],
            "candidate_output_delta_e_oklab": evaluated["candidate_output_delta_e_oklab"],
            "candidate_new_exact_boundary_fraction": evaluated["candidate_new_exact_boundary_fraction"],
            "candidate_p999_gradient_ratio": evaluated["candidate_p999_gradient_ratio"],
        })
    rows.sort(key=lambda row: row["scene_id"])
    metrics, gates = _summarize(rows, config, operator)
    passed = all(gates.values())
    implementation_paths = [
        "src/eval/repid_paired_style_retrieval.py",
        "src/eval/repid_paired_style_latent.py",
        "src/eval/repid_reference_operator_predictor.py",
        "scripts/run_u5_r2repid14_paired_style_case_routing.py",
    ]
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_bytes),
        "implementation_sha256": {path: _sha256((ROOT / path).read_bytes()) for path in implementation_paths},
        "fit_scene_count": len(fit_rows),
        "calibration_scene_count": len(calibration_rows),
        "calibration_original_reads_before_prediction_freeze": 0,
        "sealed_requests": 0,
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "model_identity_sha256": _canonical_sha(_jsonable(model)),
        "fit_label_identity_sha256": _canonical_sha([{"scene_id": item[0], "parameters": item[3].tolist(), "diagnostics": item[4]} for item in fit_payload]),
        "medoid_scene_id": fit_ids[int(model["medoid_index"])],
        "rows": rows,
        "metrics": metrics,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = f"sha256:{_canonical_sha(report)}"
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
    result = run(args.config, args.original_root, args.original_report, args.report, reverse=args.reverse)
    print(json.dumps({"report_sha256": result["sha256"], "stable_evidence_id": result["report"]["stable_evidence_id"], "decision": result["report"]["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
