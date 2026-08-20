#!/usr/bin/env python3
"""Run the frozen REPID paired-supervised after-only style-latent D0."""

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

from src.eval.repid_paired_style_latent import (
    encode_pca,
    predict_factorized,
    predict_ridge,
    spatial_descriptor,
    train_factorized_models,
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


def _rate(values: np.ndarray) -> float:
    return float(np.mean(values > 0.0))


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    fields = {
        "candidate": np.asarray([row["candidate_improvement"] for row in rows]),
        "direct": np.asarray([row["gain_vs_direct"] for row in rows]),
        "global": np.asarray([row["gain_vs_global"] for row in rows]),
        "permuted": np.asarray([row["gain_vs_permuted"] for row in rows]),
        "paired_oracle": np.asarray([row["paired_latent_oracle_improvement"] for row in rows]),
    }
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": _rate(fields["candidate"]),
        "candidate_improvement_median": float(np.median(fields["candidate"])),
        "candidate_improvement_worst": float(np.min(fields["candidate"])),
        "beat_direct_rate": _rate(fields["direct"]),
        "gain_vs_direct_median": float(np.median(fields["direct"])),
        "beat_global_rate": _rate(fields["global"]),
        "gain_vs_global_median": float(np.median(fields["global"])),
        "beat_permuted_rate": _rate(fields["permuted"]),
        "gain_vs_permuted_median": float(np.median(fields["permuted"])),
        "paired_latent_oracle_improvement_rate": _rate(fields["paired_oracle"]),
        "paired_latent_oracle_improvement_median": float(np.median(fields["paired_oracle"])),
        "median_output_delta_e_oklab": float(np.median([row["candidate_output_delta_e_oklab"] for row in rows])),
        "maximum_new_exact_boundary_fraction": float(max(row["candidate_new_exact_boundary_fraction"] for row in rows)),
        "maximum_p999_gradient_ratio": float(max(row["candidate_p999_gradient_ratio"] for row in rows)),
        "minimum_determinant": float(min(row["candidate_matrix"]["determinant"] for row in rows)),
        "maximum_condition_number": float(max(row["candidate_matrix"]["condition_number"] for row in rows)),
        "minimum_singular_value": float(min(row["candidate_matrix"]["minimum_singular_value"] for row in rows)),
        "predicted_latent_rmse": float(np.sqrt(np.mean([row["latent_squared_error"] for row in rows]))),
    }
    spec = config["calibration_gates"]
    matrix_spec = config["operator"]["matrix_gates"]
    gates = {
        "candidate_improvement_rate": metrics["candidate_improvement_rate"] >= spec["candidate_improvement_rate_min"],
        "candidate_improvement_median": metrics["candidate_improvement_median"] >= spec["candidate_improvement_median_min"],
        "candidate_improvement_worst": metrics["candidate_improvement_worst"] >= spec["candidate_improvement_worst_min"],
        "beat_direct_rate": metrics["beat_direct_rate"] >= spec["beat_direct_rate_min"],
        "gain_vs_direct_median": metrics["gain_vs_direct_median"] >= spec["gain_vs_direct_median_min"],
        "beat_global_rate": metrics["beat_global_rate"] >= spec["beat_global_rate_min"],
        "gain_vs_global_median": metrics["gain_vs_global_median"] >= spec["gain_vs_global_median_min"],
        "beat_permuted_rate": metrics["beat_permuted_rate"] >= spec["beat_permuted_rate_min"],
        "gain_vs_permuted_median": metrics["gain_vs_permuted_median"] >= spec["gain_vs_permuted_median_min"],
        "paired_latent_oracle_improvement_rate": metrics["paired_latent_oracle_improvement_rate"] >= spec["paired_latent_oracle_improvement_rate_min"],
        "paired_latent_oracle_improvement_median": metrics["paired_latent_oracle_improvement_median"] >= spec["paired_latent_oracle_improvement_median_min"],
        "style_delta": metrics["median_output_delta_e_oklab"] >= spec["median_output_delta_e_oklab_min"],
        "new_boundary": metrics["maximum_new_exact_boundary_fraction"] <= spec["new_exact_boundary_fraction_max"],
        "gradient": metrics["maximum_p999_gradient_ratio"] <= spec["p999_gradient_ratio_max"],
        "matrix_safety": metrics["minimum_determinant"] >= matrix_spec["determinant_min"] and metrics["maximum_condition_number"] <= matrix_spec["condition_number_max"] and metrics["minimum_singular_value"] >= matrix_spec["minimum_singular_value"],
    }
    return metrics, gates


def run(config_path: Path, original_root: Path, original_report_path: Path, report_path: Path, *, reverse: bool) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    rendered_bytes = (ROOT / config["parents"]["rendered_acquisition_path"]).read_bytes()
    original_evidence_bytes = (ROOT / config["parents"]["original_acquisition_evidence_path"]).read_bytes()
    repid9c_bytes = (ROOT / config["parents"]["repid9c_evidence_path"]).read_bytes()
    original_report_bytes = original_report_path.read_bytes()
    if _sha256(rendered_bytes) != config["parents"]["rendered_acquisition_sha256"]:
        raise ValueError("rendered acquisition drift")
    if _sha256(original_evidence_bytes) != config["parents"]["original_acquisition_evidence_sha256"]:
        raise ValueError("original acquisition evidence drift")
    if _sha256(original_report_bytes) != config["parents"]["original_acquisition_sha256"]:
        raise ValueError("original acquisition drift")
    if _sha256(repid9c_bytes) != config["parents"]["repid9c_evidence_sha256"] or json.loads(repid9c_bytes)["decision"] != config["parents"]["repid9c_required_decision"]:
        raise ValueError("REPID9C parent drift")
    if json.loads(original_evidence_bytes)["decision"] != "open_repid_reference_only_bounded_parameter_predictor_d0":
        raise ValueError("original acquisition is not admitted")
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

    fit_payload = []
    for row in fit_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        record = originals[scene_id]
        original = load_original_srgb(original_root / record["logical_path"], expected_sha256=record["sha256"])
        parameters, diagnostics = fit_scene_effect(original, winner, scene_id, config["operator"])
        fit_payload.append((scene_id, spatial_descriptor(winner, config["descriptor"]), spatial_descriptor(original, config["descriptor"]), parameters, diagnostics))
    fit_payload.sort(key=lambda item: item[0])
    after_features = np.stack([item[1] for item in fit_payload])
    paired_deltas = np.stack([item[1] - item[2] for item in fit_payload])
    parameters = np.stack([item[3] for item in fit_payload])
    latent_spec = config["latent"]
    models = train_factorized_models(after_features, paired_deltas, parameters, component_count=int(latent_spec["principal_components"]), student_alpha=float(latent_spec["student_ridge_alpha"]), operator_alpha=float(latent_spec["operator_ridge_alpha"]), cyclic_shift=int(config["controls"]["cyclic_shift"]))

    calibration_references = {}
    calibration_features = {}
    for row in calibration_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        calibration_references[scene_id] = winner
        calibration_features[scene_id] = spatial_descriptor(winner, config["descriptor"])
    calibration_ids = sorted(calibration_features)
    feature_matrix = np.stack([calibration_features[scene_id] for scene_id in calibration_ids])
    predictions = predict_factorized(models, feature_matrix)
    freeze_payload = {
        "scene_ids": calibration_ids,
        "candidate_latents": predictions["candidate_latent"].tolist(),
        "candidate_parameters": predictions["candidate_parameters"].tolist(),
        "direct_parameters": predictions["direct_parameters"].tolist(),
        "permuted_parameters": predictions["permuted_parameters"].tolist(),
        "global_parameters": np.asarray(models["global_parameters"]).tolist(),
    }
    prediction_freeze_sha256 = _canonical_sha(freeze_payload)

    rows = []
    for index, scene_id in enumerate(calibration_ids):
        record = originals[scene_id]
        original = load_original_srgb(original_root / record["logical_path"], expected_sha256=record["sha256"])
        winner = calibration_references[scene_id]
        actual_delta = spatial_descriptor(winner, config["descriptor"]) - spatial_descriptor(original, config["descriptor"])
        oracle_latent = encode_pca(models["pca"], actual_delta)[0]
        oracle_parameters = predict_ridge(models["latent_to_operator"], oracle_latent)[0]
        decoded = {}
        clip_counts = {}
        encoded_parameters = {
            "candidate": predictions["candidate_parameters"][index],
            "direct": predictions["direct_parameters"][index],
            "global": models["global_parameters"],
            "permuted": predictions["permuted_parameters"][index],
            "paired_latent_oracle": oracle_parameters,
        }
        for name, values in encoded_parameters.items():
            decoded[name], clip_counts[name] = decode_safe_effect(values, config["operator"])
        evaluated = evaluate_effects(original, winner, decoded)
        errors = evaluated["errors"]
        identity = evaluated["identity_error"]
        rows.append({
            "scene_id": scene_id,
            "candidate_improvement": relative_gain(identity, errors["candidate"]),
            "gain_vs_direct": relative_gain(errors["direct"], errors["candidate"]),
            "gain_vs_global": relative_gain(errors["global"], errors["candidate"]),
            "gain_vs_permuted": relative_gain(errors["permuted"], errors["candidate"]),
            "paired_latent_oracle_improvement": relative_gain(identity, errors["paired_latent_oracle"]),
            "identity_error": identity,
            "errors": errors,
            "candidate_matrix": matrix_diagnostics(decoded["candidate"]),
            "candidate_dose": decoded["candidate"].dose,
            "parameter_clip_counts": clip_counts,
            "latent_squared_error": float(np.mean(np.square(predictions["candidate_latent"][index] - oracle_latent))),
            "candidate_output_delta_e_oklab": evaluated["candidate_output_delta_e_oklab"],
            "candidate_new_exact_boundary_fraction": evaluated["candidate_new_exact_boundary_fraction"],
            "candidate_p999_gradient_ratio": evaluated["candidate_p999_gradient_ratio"],
        })
    rows.sort(key=lambda row: row["scene_id"])
    metrics, gates = _summarize(rows, config)
    passed = all(gates.values())
    implementation_paths = [
        "src/eval/repid_paired_style_latent.py",
        "src/eval/repid_reference_operator_predictor.py",
        "scripts/run_u5_r2repid13_paired_style_latent.py",
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
        "model_identity_sha256": _canonical_sha(_jsonable(models)),
        "fit_label_identity_sha256": _canonical_sha([{"scene_id": item[0], "parameters": item[3].tolist(), "diagnostics": item[4]} for item in fit_payload]),
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
