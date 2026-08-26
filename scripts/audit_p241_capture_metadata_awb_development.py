"""Run the frozen P241 development-only capture-metadata AWB discriminant."""

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

from src.eval.capture_metadata_awb import (
    angular_errors_degrees,
    apply_ridge,
    capture_features,
    fit_ridge,
    log_target_to_neutral,
    neutral_log_target,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _relative_reduction(candidate: np.ndarray, control: np.ndarray) -> np.ndarray:
    return (control - candidate) / np.maximum(control, 1e-12)


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "P241":
        raise ValueError("unexpected P241 contract")
    source = config["development_source"]
    report_path = ROOT / source["report_path"]
    if _sha256(report_path) != source["report_sha256"]:
        raise ValueError("P240 development report hash mismatch")
    p240 = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [row for row in p240["rows"] if row["complete"]]
    rows.sort(key=lambda row: int(row["id"]), reverse=reverse)
    if len(rows) != int(source["eligible_rows"]):
        raise ValueError("P240 eligible row count mismatch")

    features = np.stack([capture_features(row["facts"]) for row in rows])
    targets = np.stack([neutral_log_target(row["facts"]) for row in rows])
    target_neutrals = log_target_to_neutral(targets)
    model_config = config["models"]
    row_results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        train_indices = np.asarray(
            [item for item in range(len(rows)) if item != index], dtype=np.int64
        )
        train_indices = train_indices[
            np.argsort([int(rows[item]["id"]) for item in train_indices])
        ]
        errors: dict[str, float] = {}
        for name in ("full", "time_only", "exposure_only"):
            specification = model_config[name]
            model = fit_ridge(
                features[train_indices],
                targets[train_indices],
                feature_indices=specification["feature_indices"],
                alpha=float(specification["ridge_alpha"]),
            )
            prediction = log_target_to_neutral(
                apply_ridge(model, features[index : index + 1])
            )
            errors[name] = float(
                angular_errors_degrees(
                    prediction, target_neutrals[index : index + 1]
                )[0]
            )

        cyclic_specification = model_config["cyclic_target"]
        cyclic_targets = targets[np.roll(train_indices, -1)]
        cyclic_model = fit_ridge(
            features[train_indices],
            cyclic_targets,
            feature_indices=cyclic_specification["feature_indices"],
            alpha=float(cyclic_specification["ridge_alpha"]),
        )
        cyclic_prediction = log_target_to_neutral(
            apply_ridge(cyclic_model, features[index : index + 1])
        )
        errors["cyclic_target"] = float(
            angular_errors_degrees(
                cyclic_prediction, target_neutrals[index : index + 1]
            )[0]
        )
        global_prediction = np.mean(targets[train_indices], axis=0, keepdims=True)
        errors["global_mean"] = float(
            angular_errors_degrees(
                log_target_to_neutral(global_prediction),
                target_neutrals[index : index + 1],
            )[0]
        )
        row_results.append(
            {
                "angular_error_degrees": errors,
                "id": int(row["id"]),
                "raw_sha256": row["raw_sha256"],
            }
        )

    row_results.sort(key=lambda row: row["id"])
    names = ("full", "time_only", "exposure_only", "cyclic_target", "global_mean")
    arrays = {
        name: np.asarray(
            [row["angular_error_degrees"][name] for row in row_results],
            dtype=np.float64,
        )
        for name in names
    }
    summaries = {
        name: {
            "median_angular_error_degrees": float(np.median(values)),
            "p95_angular_error_degrees": float(np.quantile(values, 0.95)),
            "worst_angular_error_degrees": float(np.max(values)),
        }
        for name, values in arrays.items()
    }
    comparisons: dict[str, dict[str, float]] = {}
    for control in ("time_only", "exposure_only", "cyclic_target", "global_mean"):
        comparisons[control] = {
            "full_median_relative_reduction": float(
                np.median(_relative_reduction(arrays["full"], arrays[control]))
            ),
            "full_win_rate": float(np.mean(arrays["full"] < arrays[control])),
        }

    gates_config = config["evaluation"]["gates"]
    gates = {
        "fresh_exif_reads_zero": True,
        "p95_angular_error": summaries["full"]["p95_angular_error_degrees"]
        <= gates_config["maximum_p95_angular_error_degrees"],
        "raw_downloads_zero": True,
        "row_count": len(row_results) >= gates_config["minimum_rows"],
        "pixel_decodes_zero": True,
    }
    for control, comparison in comparisons.items():
        gates[f"median_reduction_vs_{control}"] = comparison[
            "full_median_relative_reduction"
        ] >= gates_config["minimum_median_relative_reduction_vs_each_control"]
        gates[f"win_rate_vs_{control}"] = comparison["full_win_rate"] >= gates_config[
            "minimum_win_rate_vs_each_control"
        ]
    passed = all(gates.values())
    return {
        "schema": "neuro-film.p241-capture-metadata-awb-development-report.v1",
        "experiment_id": "P241",
        "status": (
            "PASS_DEVELOPMENT_OPEN_SEPARATELY_FROZEN_METADATA_CONFIRMATION"
            if passed
            else "FAIL_CLOSED_BEFORE_FRESH_METADATA_CONFIRMATION"
        ),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "p240_report_sha256": _sha256(report_path),
        },
        "comparisons": comparisons,
        "gates": gates,
        "inventory": {
            "development_rows": len(row_results),
            "fresh_exif_reads": 0,
            "fresh_target_reads": 0,
            "pixel_decodes": 0,
            "raw_bytes_downloaded": 0,
        },
        "rows": row_results,
        "summaries": summaries,
        "decision": (
            "open_separately_frozen_file_model_isolated_metadata_confirmation"
            if passed
            else "close_exact_capture_feature_ridge_family_without_fresh_target_read"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))


if __name__ == "__main__":
    main()
