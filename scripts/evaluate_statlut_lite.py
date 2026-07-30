#!/usr/bin/env python3
"""Evaluate a generated-data Lab-statistics residual ridge mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match import canonical_sha256  # noqa: E402
from src.color_match.research import (  # noqa: E402
    StatLUTFeaturePolicy,
    aggregate_statlut_features,
    extract_statlut_lab_features,
)
from src.inference import atomic_write_json  # noqa: E402
from src.roll2film.synthetic_recovery import (  # noqa: E402
    SYNTHETIC_OPERATOR_SCHEMA,
    SyntheticBoundedOperator,
    generate_operator_manifest,
    manifest_sha256,
    operator_from_manifest,
    palette_cloud,
    project_operator_parameters,
)


REPORT_SCHEMA_ID = "neuro-film.statlut-lite-report.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _linear(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float64)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in (base, *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def load_contract(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / str(config["parent_config"])).read_text(encoding="utf-8")
    )
    manifest = generate_operator_manifest(parent)
    if manifest_sha256(manifest) != config["parent_manifest_sha256"]:
        raise ValueError("StatLUT-lite parent manifest hash mismatch")
    if config["execution"] != {
        "cpu_only": True,
        "generated_data_only": True,
        "neural_network_allowed": False,
        "linear_predictor_only": True,
        "model_predicts_parameters_not_rgb": True,
        "confirmation_rows_unopened_before_freeze": True,
        "product_integration_allowed": False,
    }:
        raise ValueError("StatLUT-lite execution boundary mismatch")
    return config, parent, manifest


def _policy(config: dict[str, Any]) -> StatLUTFeaturePolicy:
    values = config["feature_policy"]
    return StatLUTFeaturePolicy(
        luminance_bins=int(values["luminance_bins"]),
        chroma_bins=int(values["chroma_bins"]),
        chroma_scale=float(values["chroma_scale"]),
        chroma_gamma=float(values["chroma_gamma"]),
    )


def _rows(
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    training = config["training"]
    rows = [row for row in manifest if row["split"] == split]
    if split == training["confirmation_split"]:
        start = int(training["confirmation_within_split_start"])
        count = int(training["confirmation_operators_per_family"])
        rows = [
            row
            for row in rows
            if start
            <= int(row["within_split_index"])
            < start + count
        ]
    return rows


def _sample(
    row: dict[str, Any],
    observation: int,
    palette_names: list[str],
    pixel_count: int,
    base_seed: int,
    policy: StatLUTFeaturePolicy,
) -> dict[str, Any]:
    operator = operator_from_manifest(row)
    index = int(row["family_index"])
    source_features = []
    for palette_index, name in enumerate(palette_names):
        cloud = palette_cloud(
            name,
            pixel_count,
            _seed(base_seed, "source", index, observation, palette_index),
        )
        source_features.append(
            extract_statlut_lab_features(
                _linear(cloud).reshape(32, 32, 3),
                policy=policy,
            )
        )
    source = aggregate_statlut_features(source_features)
    reference_name = palette_names[
        (index + observation + 1) % len(palette_names)
    ]
    query_name = palette_names[
        (index + observation) % len(palette_names)
    ]
    reference_neutral = palette_cloud(
        reference_name,
        pixel_count,
        _seed(base_seed, "reference", index, observation),
    )
    styled_reference = operator.apply(reference_neutral)
    style = extract_statlut_lab_features(
        _linear(styled_reference).reshape(32, 32, 3),
        policy=policy,
    ).vector()
    query = palette_cloud(
        query_name,
        pixel_count,
        _seed(base_seed, "query", index, observation),
    )
    feature = np.concatenate((source, style, style - source))
    return {
        "operator_id": operator.operator_id,
        "truth_parameters": operator.parameters,
        "feature": feature,
        "query": query,
        "target": operator.apply(query),
    }


def build_samples(
    config: dict[str, Any],
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    palette_names = (
        list(parent["palettes"]["stress"])
        if split == config["training"]["confirmation_split"]
        else list(parent["palettes"]["development"])
    )
    return [
        _sample(
            row,
            observation,
            palette_names,
            int(parent["palettes"]["pixels_per_cloud"]),
            _seed(int(config["training"]["seed"]), split),
            _policy(config),
        )
        for row in rows
        for observation in range(
            int(config["training"]["observations_per_operator"])
        )
    ]


def _fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x = (features - mean) / scale
    target_mean = targets.mean(axis=0)
    y = targets - target_mean
    dual = np.linalg.solve(
        x @ x.T + float(alpha) * np.eye(len(x)),
        y,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": x.T @ dual,
        "target_mean": target_mean,
    }


def _predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = (features - model["mean"]) / model["scale"]
    raw = x @ model["coefficient"] + model["target_mean"]
    return np.stack(
        [project_operator_parameters(row) for row in raw],
        axis=0,
    )


def _operator(parameters: np.ndarray, label: str) -> SyntheticBoundedOperator:
    return SyntheticBoundedOperator(
        parameters=parameters,
        family="combined",
        operator_id=label,
    )


def _grid(axis_size: int = 9) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def _evaluate(
    samples: list[dict[str, Any]],
    predictions: np.ndarray,
    global_parameters: np.ndarray,
) -> dict[str, Any]:
    grid = _grid()
    records = []
    for sample, parameters in zip(samples, predictions):
        truth = _operator(sample["truth_parameters"], "truth")
        predicted = _operator(parameters, "predicted")
        global_mean = _operator(global_parameters, "global")
        truth_grid = truth.apply(grid)
        predicted_grid = predicted.apply(grid)
        global_grid = global_mean.apply(grid)
        query = sample["query"]
        target = sample["target"]
        predicted_output = predicted.apply(query)
        global_output = global_mean.apply(query)
        identity_rmse = float(np.sqrt(np.mean((query - target) ** 2)))
        predicted_rmse = float(
            np.sqrt(np.mean((predicted_output - target) ** 2))
        )
        global_rmse = float(
            np.sqrt(np.mean((global_output - target) ** 2))
        )
        records.append(
            {
                "operator_id": sample["operator_id"],
                "operator_grid_rmse": float(
                    np.sqrt(np.mean((predicted_grid - truth_grid) ** 2))
                ),
                "global_grid_rmse": float(
                    np.sqrt(np.mean((global_grid - truth_grid) ** 2))
                ),
                "captured_style_fraction": (
                    (identity_rmse - predicted_rmse)
                    / max(identity_rmse, 1e-12)
                ),
                "global_captured_style_fraction": (
                    (identity_rmse - global_rmse)
                    / max(identity_rmse, 1e-12)
                ),
            }
        )
    grid_errors = np.asarray(
        [record["operator_grid_rmse"] for record in records]
    )
    captured = np.asarray(
        [record["captured_style_fraction"] for record in records]
    )
    global_grid = np.asarray(
        [record["global_grid_rmse"] for record in records]
    )
    return {
        "summary": {
            "observations": len(records),
            "operator_grid_rmse_median": float(np.median(grid_errors)),
            "operator_grid_rmse_p90": float(np.quantile(grid_errors, 0.9)),
            "captured_style_fraction_median": float(np.median(captured)),
            "improved_over_identity_fraction": float(np.mean(captured > 0.0)),
            "grid_improvement_over_global_mean_fraction": float(
                (np.median(global_grid) - np.median(grid_errors))
                / max(float(np.median(global_grid)), 1e-12)
            ),
        },
        "records": records,
    }


def _replicate_error(
    samples: list[dict[str, Any]],
    predictions: np.ndarray,
) -> float:
    grid = _grid()
    grouped: dict[str, list[np.ndarray]] = {}
    for sample, parameters in zip(samples, predictions):
        grouped.setdefault(sample["operator_id"], []).append(parameters)
    errors = []
    for values in grouped.values():
        first = _operator(values[0], "a").apply(grid)
        second = _operator(values[1], "b").apply(grid)
        errors.append(float(np.sqrt(np.mean((first - second) ** 2))))
    return float(np.median(errors))


def main() -> int:
    args = _parser().parse_args()
    config, parent, manifest = load_contract(args.config)
    fit = build_samples(
        config,
        parent,
        _rows(config, manifest, config["training"]["fit_split"]),
        config["training"]["fit_split"],
    )
    validation = build_samples(
        config,
        parent,
        _rows(config, manifest, config["training"]["selection_split"]),
        config["training"]["selection_split"],
    )
    confirmation = build_samples(
        config,
        parent,
        _rows(config, manifest, config["training"]["confirmation_split"]),
        config["training"]["confirmation_split"],
    )
    x_fit = np.stack([sample["feature"] for sample in fit])
    y_fit = np.stack([sample["truth_parameters"] for sample in fit])
    x_validation = np.stack([sample["feature"] for sample in validation])
    global_parameters = project_operator_parameters(y_fit.mean(axis=0))
    selections = []
    models = {}
    for alpha in config["training"]["ridge_alphas"]:
        model = _fit_ridge(x_fit, y_fit, float(alpha))
        models[float(alpha)] = model
        prediction = _predict(model, x_validation)
        result = _evaluate(validation, prediction, global_parameters)
        selections.append(
            {
                "alpha": float(alpha),
                "operator_grid_rmse_median": result["summary"][
                    "operator_grid_rmse_median"
                ],
            }
        )
    selections.sort(
        key=lambda row: (row["operator_grid_rmse_median"], row["alpha"])
    )
    selected_alpha = selections[0]["alpha"]
    model = models[selected_alpha]
    x_confirmation = np.stack(
        [sample["feature"] for sample in confirmation]
    )
    prediction = _predict(model, x_confirmation)
    result = _evaluate(confirmation, prediction, global_parameters)
    replicate = _replicate_error(confirmation, prediction)

    identity_parameters = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75]
    )
    identity_manifest = []
    for index, row in enumerate(
        _rows(
            config,
            manifest,
            config["training"]["confirmation_split"],
        )[:12]
    ):
        copied = dict(row)
        copied["schema"] = SYNTHETIC_OPERATOR_SCHEMA
        copied["operator_id"] = f"identity-{index:04d}"
        copied["family"] = "combined"
        copied["parameters"] = identity_parameters.tolist()
        identity_manifest.append(copied)
    identity_rows = build_samples(
        config,
        parent,
        identity_manifest,
        config["training"]["confirmation_split"],
    )
    identity_prediction = _predict(
        model,
        np.stack([sample["feature"] for sample in identity_rows]),
    )
    identity_result = _evaluate(
        identity_rows,
        identity_prediction,
        global_parameters,
    )
    identity_max = max(
        record["operator_grid_rmse"]
        for record in identity_result["records"]
    )
    summary = result["summary"]
    gates = config["gates"]
    checks = {
        "median_grid": summary["operator_grid_rmse_median"]
        <= float(gates["maximum_operator_grid_rmse_median"]),
        "p90_grid": summary["operator_grid_rmse_p90"]
        <= float(gates["maximum_operator_grid_rmse_p90"]),
        "captured_style": summary["captured_style_fraction_median"]
        >= float(gates["minimum_captured_style_fraction_median"]),
        "global_mean": summary[
            "grid_improvement_over_global_mean_fraction"
        ]
        >= float(gates["minimum_improvement_over_global_mean_fraction"]),
        "same_look": replicate
        <= float(gates["maximum_same_look_replicate_grid_rmse_median"]),
        "identity": identity_max
        <= float(gates["maximum_identity_reference_grid_rmse"]),
    }
    decision = {
        "status": (
            "statlut-lite-confirmation-passed"
            if all(checks.values())
            else "statlut-lite-linear-route-closed"
        ),
        "checks": checks,
        "product_integration_open": False,
        "nonlinear_mapper_open": all(
            checks[key]
            for key in ("same_look", "identity")
        )
        and not all(checks.values()),
    }
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "experiment_id": config["experiment_id"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "feature_dimension": int(x_fit.shape[1]),
        "fit_observations": len(fit),
        "validation_observations": len(validation),
        "confirmation_observations": len(confirmation),
        "alpha_selection": selections,
        "selected_alpha": selected_alpha,
        "confirmation": result,
        "same_look_replicate_grid_rmse_median": replicate,
        "identity_reference_grid_rmse_maximum": identity_max,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["report_id"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    report_sha256 = atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "report_id": payload["report_id"],
                "report_sha256": report_sha256,
                "selected_alpha": selected_alpha,
                "decision": decision,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
