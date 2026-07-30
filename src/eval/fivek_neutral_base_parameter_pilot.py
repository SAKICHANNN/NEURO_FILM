"""Bounded explicit FiveK neutral-base parameter prediction pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.build_fivek_freeze_pack import (
    load_expert_icc_srgb,
    load_raw_default,
    filtered_target,
    resize_to_shape,
)


class FiveKNeutralBasePilotError(ValueError):
    """Raised when the parameter-pilot contract or evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def validate_contract(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKNeutralBasePilotError("pilot contract is not frozen")
    evidence = config["source_evidence"]
    for key in ("manifest", "report"):
        path = root / str(evidence[key])
        if not path.is_file() or _sha256(path) != str(
            evidence[f"{key}_sha256"]
        ):
            raise FiveKNeutralBasePilotError(f"source evidence drift: {key}")
    report = json.loads(
        (root / str(evidence["report"])).read_text(encoding="utf-8")
    )
    if report["automatic_pass"] is not evidence["required_automatic_pass"]:
        raise FiveKNeutralBasePilotError("source evidence is not passing")
    operator = config["operator"]
    if operator["direct_final_rgb_learning_forbidden"] is not True:
        raise FiveKNeutralBasePilotError("final-RGB prohibition missing")
    if len(operator["parameter_order"]) != 5:
        raise FiveKNeutralBasePilotError("parameter schema drift")
    if config["descriptor"]["semantic_features_forbidden"] is not True:
        raise FiveKNeutralBasePilotError("semantic-feature prohibition missing")


def apply_neutral_base(
    encoded_rgb: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Apply the explicit cube-safe neutral-base operator."""

    source = np.asarray(encoded_rgb, dtype=np.float64)
    params = np.asarray(parameters, dtype=np.float64)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or params.shape != (5,)
        or not np.all(np.isfinite(params))
    ):
        raise FiveKNeutralBasePilotError(
            "source/parameters must be finite bounded RGB and five scalars"
        )
    if np.array_equal(
        params, np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    ):
        return source.copy()
    contrast, exposure, red_wb, blue_wb, saturation = params
    if (
        not 0.0 < contrast
        or not 0.0 < saturation <= 1.0
    ):
        raise FiveKNeutralBasePilotError("invalid contrast or saturation")
    shifts = np.array(
        [exposure + red_wb, exposure, exposure + blue_wb],
        dtype=np.float64,
    )
    interior = (source > 0.0) & (source < 1.0)
    transformed = np.empty_like(source)
    transformed[source <= 0.0] = 0.0
    transformed[source >= 1.0] = 1.0
    if np.any(interior):
        clipped = np.clip(
            source[interior],
            np.finfo(np.float64).tiny,
            1.0 - np.finfo(np.float64).eps,
        )
        channel_indices = np.nonzero(interior)[-1]
        logits = np.log(clipped) - np.log1p(-clipped)
        z = contrast * logits + shifts[channel_indices]
        transformed[interior] = 1.0 / (1.0 + np.exp(-z))
    luma = (
        0.2126 * transformed[..., 0]
        + 0.7152 * transformed[..., 1]
        + 0.0722 * transformed[..., 2]
    )
    result = luma[..., None] + saturation * (
        transformed - luma[..., None]
    )
    if (
        not np.all(np.isfinite(result))
        or np.any(result < -1e-12)
        or np.any(result > 1.0 + 1e-12)
    ):
        raise FiveKNeutralBasePilotError("operator escaped the RGB cube")
    return np.clip(result, 0.0, 1.0)


def source_descriptor(
    encoded_rgb: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    source = np.asarray(encoded_rgb, dtype=np.float64)
    luma = (
        0.2126 * source[..., 0]
        + 0.7152 * source[..., 1]
        + 0.0722 * source[..., 2]
    )
    arrays = [source[..., index].ravel() for index in range(3)]
    arrays.append(luma.ravel())
    q = np.asarray(
        config["descriptor"]["channel_and_luma_quantiles"],
        dtype=np.float64,
    )
    features: list[float] = []
    for values in arrays:
        features.extend(np.quantile(values, q).tolist())
        features.extend([float(np.mean(values)), float(np.std(values))])
    chroma = np.sqrt(
        np.sum((source - luma[..., None]) ** 2, axis=2)
    ).ravel()
    cq = np.asarray(
        config["descriptor"]["chroma_quantiles"], dtype=np.float64
    )
    features.extend(np.quantile(chroma, cq).tolist())
    features.extend([float(np.mean(chroma)), float(np.std(chroma))])
    return np.asarray(features, dtype=np.float64)


def _fit_parameters(
    source: np.ndarray,
    target: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, bool]:
    operator = config["operator"]
    fit = config["fit"]
    stride = int(fit["sample_stride"])
    sampled_source = source[::stride, ::stride]
    sampled_target = target[::stride, ::stride]
    identity = np.asarray(operator["identity"], dtype=np.float64)
    spans = np.asarray(operator["upper_bounds"], dtype=np.float64) - np.asarray(
        operator["lower_bounds"], dtype=np.float64
    )
    regularization = float(fit["parameter_regularization"])

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = apply_neutral_base(sampled_source, parameters)
        pixel = (predicted.astype(np.float64) - sampled_target).ravel()
        penalty = (
            np.sqrt(regularization)
            * (parameters - identity)
            / spans
        )
        return np.concatenate([pixel, penalty])

    result = least_squares(
        residual,
        x0=identity,
        bounds=(
            np.asarray(operator["lower_bounds"], dtype=np.float64),
            np.asarray(operator["upper_bounds"], dtype=np.float64),
        ),
        method="trf",
        max_nfev=int(fit["maximum_evaluations"]),
    )
    return np.asarray(result.x, dtype=np.float64), bool(result.success)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray
) -> float:
    source_boundary = (source <= 0.0) | (source >= 1.0)
    output_boundary = (output <= 0.0) | (output >= 1.0)
    return float(np.mean(output_boundary & ~source_boundary))


def _parameter_predictions(
    descriptors: np.ndarray,
    parameters: np.ndarray,
    groups: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    ridge_predictions = np.empty_like(parameters)
    global_predictions = np.empty_like(parameters)
    nearest_predictions = np.empty_like(parameters)
    folds: list[dict[str, Any]] = []
    outer = GroupKFold(n_splits=5)
    bounds_low = np.asarray(
        config["operator"]["lower_bounds"], dtype=np.float64
    )
    bounds_high = np.asarray(
        config["operator"]["upper_bounds"], dtype=np.float64
    )
    alphas = [float(value) for value in config["evaluation"]["ridge_alphas"]]
    for fold_index, (train, test) in enumerate(
        outer.split(descriptors, parameters, groups), start=1
    ):
        scaler = StandardScaler().fit(descriptors[train])
        train_x = scaler.transform(descriptors[train])
        test_x = scaler.transform(descriptors[test])
        train_groups = groups[train]
        inner_splits = min(4, len(np.unique(train_groups)))
        alpha_scores: dict[float, float] = {}
        for alpha in alphas:
            errors = []
            inner = GroupKFold(n_splits=inner_splits)
            for inner_train, inner_valid in inner.split(
                train_x, parameters[train], train_groups
            ):
                model = Ridge(alpha=alpha).fit(
                    train_x[inner_train], parameters[train][inner_train]
                )
                predicted = model.predict(train_x[inner_valid])
                errors.append(
                    float(
                        np.mean(
                            (
                                (predicted - parameters[train][inner_valid])
                                / (bounds_high - bounds_low)
                            )
                            ** 2
                        )
                    )
                )
            alpha_scores[alpha] = float(np.mean(errors))
        selected_alpha = min(alphas, key=lambda value: (alpha_scores[value], value))
        model = Ridge(alpha=selected_alpha).fit(train_x, parameters[train])
        ridge_predictions[test] = np.clip(
            model.predict(test_x), bounds_low, bounds_high
        )
        global_predictions[test] = np.median(parameters[train], axis=0)
        distances = np.sum(
            (test_x[:, None, :] - train_x[None, :, :]) ** 2, axis=2
        )
        nearest_predictions[test] = parameters[
            train[np.argmin(distances, axis=1)]
        ]
        folds.append(
            {
                "fold": fold_index,
                "train_count": int(len(train)),
                "test_count": int(len(test)),
                "train_group_count": int(len(np.unique(groups[train]))),
                "test_groups": sorted(str(value) for value in np.unique(groups[test])),
                "selected_alpha": selected_alpha,
                "inner_scores": {
                    str(key): alpha_scores[key] for key in sorted(alpha_scores)
                },
            }
        )
    return ridge_predictions, global_predictions, nearest_predictions, folds


def _bootstrap_lower(
    global_errors: np.ndarray,
    ridge_errors: np.ndarray,
    config: Mapping[str, Any],
) -> float:
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    count = int(config["evaluation"]["bootstrap_resamples"])
    indices = rng.integers(
        0, len(global_errors), size=(count, len(global_errors))
    )
    improvements = np.mean(
        global_errors[indices] - ridge_errors[indices], axis=1
    )
    return float(np.quantile(improvements, 0.025))


def run_pilot(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    manifest = json.loads(
        (
            root / str(config["source_evidence"]["manifest"])
        ).read_text(encoding="utf-8")
    )
    rows = manifest["rows"]
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    fitted: list[np.ndarray] = []
    fit_success: list[bool] = []
    identity_errors: list[float] = []
    oracle_errors: list[float] = []
    target_policy = type(
        "TargetPolicy",
        (),
        {
            "luma_strength": config["neutral_target"]["luma_strength"],
            "chroma_strength": config["neutral_target"]["chroma_strength"],
            "chroma_headroom": config["neutral_target"]["chroma_headroom"],
            "wb_anchor_strength": config["neutral_target"][
                "white_balance_anchor_strength"
            ],
        },
    )()
    for row in rows:
        raw, _ = load_raw_default(
            root / row["raw_path"], int(config["decode"]["maximum_side"])
        )
        raw = np.clip(raw, 0.0, 1.0)
        expert = load_expert_icc_srgb(
            root / row["expert_path"], int(config["decode"]["maximum_side"])
        )
        if expert.shape != raw.shape:
            expert = resize_to_shape(expert, raw.shape[:2])
        expert = np.clip(expert, 0.0, 1.0)
        target = filtered_target(raw, expert, target_policy)
        parameters, success = _fit_parameters(raw, target, config)
        oracle = apply_neutral_base(raw, parameters)
        sources.append(raw)
        targets.append(target)
        descriptors.append(source_descriptor(raw, config))
        fitted.append(parameters)
        fit_success.append(success)
        identity_errors.append(_rmse(raw, target))
        oracle_errors.append(_rmse(oracle, target))
    descriptor_array = np.stack(descriptors)
    parameter_array = np.stack(fitted)
    groups = np.asarray([row["camera_group_id"] for row in rows])
    ridge, global_params, nearest, folds = _parameter_predictions(
        descriptor_array, parameter_array, groups, config
    )
    method_params = {
        "global": global_params,
        "nearest": nearest,
        "ridge": ridge,
    }
    errors: dict[str, list[float]] = {
        "identity": identity_errors,
        "oracle": oracle_errors,
        "global": [],
        "nearest": [],
        "ridge": [],
    }
    boundary: dict[str, list[float]] = {
        "global": [],
        "nearest": [],
        "ridge": [],
    }
    per_row: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = {
            "pair_id": row["pair_id"],
            "camera_group_id": row["camera_group_id"],
            "fit_success": fit_success[index],
            "fitted_parameters": fitted[index].tolist(),
            "identity_rmse": identity_errors[index],
            "oracle_rmse": oracle_errors[index],
        }
        for method, predictions in method_params.items():
            output = apply_neutral_base(sources[index], predictions[index])
            value = _rmse(output, targets[index])
            boundary_value = _new_boundary_fraction(
                sources[index], output
            )
            errors[method].append(value)
            boundary[method].append(boundary_value)
            record[f"{method}_parameters"] = predictions[index].tolist()
            record[f"{method}_rmse"] = value
            record[f"{method}_new_boundary_fraction"] = boundary_value
        per_row.append(record)
    metrics: dict[str, Any] = {}
    for method, values in errors.items():
        array = np.asarray(values, dtype=np.float64)
        metrics[method] = {
            "mean_rmse": float(np.mean(array)),
            "median_rmse": float(np.median(array)),
            "p95_rmse": float(np.quantile(array, 0.95)),
        }
    identity_array = np.asarray(errors["identity"])
    oracle_array = np.asarray(errors["oracle"])
    global_array = np.asarray(errors["global"])
    ridge_array = np.asarray(errors["ridge"])
    oracle_improvements = (identity_array - oracle_array) / np.maximum(
        identity_array, 1e-12
    )
    ridge_mean_improvement = float(
        (np.mean(global_array) - np.mean(ridge_array))
        / np.mean(global_array)
    )
    ridge_win_fraction = float(np.mean(ridge_array < global_array))
    ridge_p95_ratio = float(
        np.quantile(ridge_array, 0.95)
        / np.quantile(global_array, 0.95)
    )
    bootstrap_lower = _bootstrap_lower(global_array, ridge_array, config)
    maximum_new_boundary = max(
        max(values, default=0.0) for values in boundary.values()
    )
    observed = {
        "per_pair_oracle_median_improvement_over_identity": float(
            np.median(oracle_improvements)
        ),
        "ridge_mean_improvement_over_global": ridge_mean_improvement,
        "ridge_win_fraction_over_global": ridge_win_fraction,
        "ridge_bootstrap_improvement_lower": bootstrap_lower,
        "ridge_p95_rmse_ratio_to_global": ridge_p95_ratio,
        "maximum_new_boundary_fraction": maximum_new_boundary,
        "fit_success_fraction": float(np.mean(fit_success)),
    }
    gates_config = config["pass_gates"]
    gates = {
        "per_pair_oracle": observed[
            "per_pair_oracle_median_improvement_over_identity"
        ]
        >= float(
            gates_config[
                "per_pair_oracle_median_improvement_over_identity_minimum"
            ]
        ),
        "ridge_mean": observed["ridge_mean_improvement_over_global"]
        >= float(gates_config["ridge_mean_improvement_over_global_minimum"]),
        "ridge_wins": observed["ridge_win_fraction_over_global"]
        >= float(gates_config["ridge_win_fraction_over_global_minimum"]),
        "ridge_bootstrap": observed["ridge_bootstrap_improvement_lower"]
        > float(
            gates_config[
                "ridge_bootstrap_improvement_lower_greater_than"
            ]
        ),
        "ridge_tail": observed["ridge_p95_rmse_ratio_to_global"]
        <= float(gates_config["ridge_p95_rmse_ratio_to_global_maximum"]),
        "new_boundaries": observed["maximum_new_boundary_fraction"]
        <= float(gates_config["new_boundary_fraction_maximum"]),
        "fit_success": observed["fit_success_fraction"]
        >= float(gates_config["fit_success_fraction_minimum"]),
    }
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "source_manifest_sha256": config["source_evidence"][
            "manifest_sha256"
        ],
        "row_count": len(rows),
        "camera_group_count": int(len(np.unique(groups))),
        "parameter_order": config["operator"]["parameter_order"],
        "metrics": metrics,
        "observed": observed,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "folds": folds,
        "rows": per_row,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(_canonical_bytes(stable_payload)).hexdigest()
    report = {
        "schema_version": 1,
        "software_commit": software_commit,
        **stable_payload,
        "stable_evidence_id": stable_id,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_bytes = _canonical_bytes(report)
    (output_dir / str(config["output"]["report"])).write_bytes(report_bytes)
    return {
        "report": report,
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


__all__ = [
    "FiveKNeutralBasePilotError",
    "apply_neutral_base",
    "run_pilot",
    "source_descriptor",
    "validate_contract",
]
