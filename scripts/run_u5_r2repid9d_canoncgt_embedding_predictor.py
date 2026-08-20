#!/usr/bin/env python3
"""Run the frozen CanonCGT-embedding REPID operator-prediction D0."""

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

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    _load_model_reference_only,
    _tensor_from_array,
)
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


def _embedding(
    reference: np.ndarray, torch: Any, model: Any, device: str
) -> np.ndarray:
    with torch.inference_mode():
        tensor = _tensor_from_array(reference, torch, device)
        feature = model.Embedding_Net(tensor)[0].detach().cpu().numpy()
    values = np.asarray(feature, dtype=np.float64)
    if values.shape != (64,) or not np.all(np.isfinite(values)):
        raise ValueError("CanonCGT embedding contract drift")
    return values


def _summarize(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    improvements = np.asarray([row["candidate_improvement"] for row in rows])
    gains = {
        name: np.asarray([row[f"gain_vs_{name}"] for row in rows])
        for name in ("raw_stat", "global", "permuted")
    }
    spec = config["calibration_gates"]
    metrics = {
        "row_count": len(rows),
        "candidate_improvement_rate": float(np.mean(improvements > 0.0)),
        "candidate_improvement_median": float(np.median(improvements)),
        "candidate_improvement_worst": float(np.min(improvements)),
        **{
            f"beat_{name}_rate": float(np.mean(value > 0.0))
            for name, value in gains.items()
        },
        **{
            f"gain_vs_{name}_median": float(np.median(value))
            for name, value in gains.items()
        },
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
        >= spec["candidate_improvement_rate_min"],
        "candidate_improvement_median": metrics["candidate_improvement_median"]
        >= spec["candidate_improvement_median_min"],
        "candidate_improvement_worst": metrics["candidate_improvement_worst"]
        >= spec["candidate_improvement_worst_min"],
        **{
            f"beat_{name}_rate": metrics[f"beat_{name}_rate"]
            >= spec[f"beat_{name}_rate_min"]
            for name in gains
        },
        **{
            f"gain_vs_{name}_median": metrics[f"gain_vs_{name}_median"]
            >= spec[f"gain_vs_{name}_median_min"]
            for name in gains
        },
        "style_delta": metrics["median_output_delta_e_oklab"]
        >= spec["median_output_delta_e_oklab_min"],
        "new_boundary": metrics["maximum_new_exact_boundary_fraction"]
        <= spec["new_exact_boundary_fraction_max"],
        "gradient": metrics["maximum_p999_gradient_ratio"]
        <= spec["p999_gradient_ratio_max"],
    }
    return metrics, gates


def run(
    config_path: Path,
    original_root: Path,
    original_report_path: Path,
    closed_raw_stat_report_path: Path,
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
    original_evidence_bytes = (
        ROOT / config["parents"]["original_acquisition_evidence_path"]
    ).read_bytes()
    original_report_bytes = original_report_path.read_bytes()
    closed_report_bytes = closed_raw_stat_report_path.read_bytes()
    bindings = (
        (rendered_bytes, config["parents"]["rendered_acquisition_sha256"]),
        (
            original_evidence_bytes,
            config["parents"]["original_acquisition_evidence_sha256"],
        ),
        (original_report_bytes, config["parents"]["original_acquisition_sha256"]),
        (closed_report_bytes, config["parents"]["closed_raw_stat_report_sha256"]),
    )
    if any(_sha256(payload) != expected for payload, expected in bindings):
        raise ValueError("parent identity drift")
    if (
        json.loads(original_evidence_bytes)["decision"]
        != "open_repid_reference_only_bounded_parameter_predictor_d0"
    ):
        raise ValueError("original acquisition is not admitted")
    rendered = json.loads(rendered_bytes)
    original_report = json.loads(original_report_bytes)
    originals = {row["scene_id"]: row for row in original_report["records"]}
    winners = _winner_lookup(rendered)
    ordered = sorted(
        rendered["records"], key=lambda row: row["scene_id"], reverse=reverse
    )
    fit_rows = [row for row in ordered if row["role"] == "fit"]
    calibration_rows = [row for row in ordered if row["role"] == "calibration"]
    if (
        len(fit_rows) != config["roles"]["fit_scene_count"]
        or len(calibration_rows) != config["roles"]["calibration_scene_count"]
    ):
        raise ValueError("role drift")
    canoncgt_bytes = (ROOT / config["canoncgt"]["contract_path"]).read_bytes()
    if _sha256(canoncgt_bytes) != config["canoncgt"]["contract_sha256"]:
        raise ValueError("CanonCGT contract drift")
    canoncgt = json.loads(canoncgt_bytes)
    torch, embedding_model, _, model_audit = _load_model_reference_only(
        ROOT, canoncgt, str(config["canoncgt"]["device"])
    )
    fit_records = []
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
        fit_records.append(
            (
                scene_id,
                _embedding(
                    winner, torch, embedding_model, config["canoncgt"]["device"]
                ),
                reference_descriptor(winner, config["raw_stat_descriptor"]),
                parameters,
                diagnostics,
            )
        )
    fit_records.sort(key=lambda item: item[0])
    embedding_features = np.stack([item[1] for item in fit_records])
    raw_features = np.stack([item[2] for item in fit_records])
    targets = np.stack([item[3] for item in fit_records])
    alpha = float(config["predictor"]["ridge_alpha"])
    model = fit_ridge(embedding_features, targets, alpha=alpha)
    raw_model = fit_ridge(raw_features, targets, alpha=alpha)
    permuted_model = fit_ridge(
        embedding_features,
        np.roll(targets, int(config["controls"]["cyclic_parameter_shift"]), axis=0),
        alpha=alpha,
    )
    global_parameters = np.mean(targets, axis=0)
    calibration_references: dict[str, np.ndarray] = {}
    embedding_cal: dict[str, np.ndarray] = {}
    raw_cal: dict[str, np.ndarray] = {}
    for row in calibration_rows:
        scene_id = row["scene_id"]
        winner = _load_winner(winners[scene_id])
        calibration_references[scene_id] = winner
        embedding_cal[scene_id] = _embedding(
            winner, torch, embedding_model, config["canoncgt"]["device"]
        )
        raw_cal[scene_id] = reference_descriptor(winner, config["raw_stat_descriptor"])
    canonical_ids = sorted(embedding_cal)
    predictions = predict_ridge(
        model, np.stack([embedding_cal[key] for key in canonical_ids])
    )
    raw_predictions = predict_ridge(
        raw_model, np.stack([raw_cal[key] for key in canonical_ids])
    )
    permuted_predictions = predict_ridge(
        permuted_model, np.stack([embedding_cal[key] for key in canonical_ids])
    )
    freeze_payload = {
        "scene_ids": canonical_ids,
        "predictions": predictions.tolist(),
        "raw_predictions": raw_predictions.tolist(),
        "permuted_predictions": permuted_predictions.tolist(),
        "global_parameters": global_parameters.tolist(),
    }
    freeze_sha = _canonical_sha(freeze_payload)
    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows = []
    for index, scene_id in enumerate(canonical_ids):
        original_record = originals[scene_id]
        original = load_original_srgb(
            original_root / original_record["logical_path"],
            expected_sha256=original_record["sha256"],
        )
        winner = calibration_references[scene_id]
        parameters = {
            "candidate": predictions[index],
            "raw_stat": raw_predictions[index],
            "global": global_parameters,
            "permuted": permuted_predictions[index],
        }
        operators = {}
        clip_counts = {}
        for name, value in parameters.items():
            operators[name], clip_counts[name] = decode_safe_effect(
                value, config["operator"]
            )
        oracle_parameters, _ = fit_scene_effect(
            original, winner, scene_id, config["operator"]
        )
        operators["oracle"], _ = decode_safe_effect(
            oracle_parameters, config["operator"]
        )
        evaluated = evaluate_effects(original, winner, operators)
        identity = evaluated["identity_error"]
        errors = evaluated["errors"]
        rows.append(
            {
                "scene_id": scene_id,
                "candidate_improvement": relative_gain(identity, errors["candidate"]),
                "gain_vs_raw_stat": relative_gain(
                    errors["raw_stat"], errors["candidate"]
                ),
                "gain_vs_global": relative_gain(errors["global"], errors["candidate"]),
                "gain_vs_permuted": relative_gain(
                    errors["permuted"], errors["candidate"]
                ),
                "oracle_improvement": relative_gain(identity, errors["oracle"]),
                "identity_error": identity,
                "errors": errors,
                "candidate_matrix": matrix_diagnostics(operators["candidate"]),
                "candidate_dose": operators["candidate"].dose,
                "parameter_clip_counts": clip_counts,
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
        "model_audit": model_audit,
        "fit_scene_count": len(fit_rows),
        "calibration_scene_count": len(calibration_rows),
        "calibration_original_reads_before_prediction_freeze": 0,
        "sealed_requests": 0,
        "prediction_freeze_sha256": freeze_sha,
        "model": model_payload(model),
        "raw_stat_model": model_payload(raw_model),
        "permuted_model": model_payload(permuted_model),
        "fit_label_identity_sha256": _canonical_sha(
            [
                {
                    "scene_id": item[0],
                    "parameters": item[3].tolist(),
                    "diagnostics": item[4],
                }
                for item in fit_records
            ]
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
    parser.add_argument("--closed-raw-stat-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        args.original_root.resolve(),
        args.original_report.resolve(),
        args.closed_raw_stat_report.resolve(),
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
