"""Target-free uncertainty diagnostic for ridge-to-global fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.build_fivek_freeze_pack import load_raw_default
from src.eval.fivek_neutral_base_parameter_pilot import source_descriptor
from src.eval.fivek_unseen_content_confirmation import load_srgb16


class FiveKTargetFreeFallbackError(ValueError):
    """Raised when diagnostic evidence or target-free boundaries drift."""


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


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKTargetFreeFallbackError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKTargetFreeFallbackError("contract is not frozen")
    if (
        config.get("training_allowed")
        or config.get("new_operator_fitting_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("visual_review_allowed")
        or config["routing"].get("target_or_oracle_features_allowed")
    ):
        raise FiveKTargetFreeFallbackError("diagnostic boundary drift")
    failed = config["failed_confirmation"]
    report = _load_hashed_json(
        root, failed["report"], failed["report_sha256"]
    )
    decision = _load_hashed_json(
        root, failed["decision"], failed["decision_sha256"]
    )
    development = config["development_model"]
    development_config = _load_hashed_json(
        root, development["config"], development["config_sha256"]
    )
    development_report = _load_hashed_json(
        root, development["report"], development["report_sha256"]
    )
    if (
        report.get("automatic_pass") is not False
        or decision.get("status") != failed["required_status"]
        or failed.get("use", "").startswith("development") is not True
        or development.get("change_allowed") is not False
    ):
        raise FiveKTargetFreeFallbackError("failed-parent boundary drift")
    expected_scores = {
        "standardized_centroid_l2",
        "standardized_nearest_neighbor_l2",
        "leave_one_training_group_parameter_disagreement_l2",
        "prediction_to_global_normalized_l2",
        "prediction_bound_proximity",
        "safe_executor_limited_fraction",
    }
    if set(config["score_candidates"]) != expected_scores:
        raise FiveKTargetFreeFallbackError("score candidate drift")
    return {
        "failed_report": report,
        "development_config": development_config,
        "development_report": development_report,
    }


def _fit_training_state(
    *,
    root: Path,
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    maximum_side: int,
) -> dict[str, Any]:
    development_config = validated["development_config"]
    source_manifest = json.loads(
        (
            root
            / development_config["source_evidence"]["manifest"]
        ).read_text(encoding="utf-8")
    )
    source_by_id = {
        row["pair_id"]: row for row in source_manifest["rows"]
    }
    descriptors = []
    parameters = []
    groups = []
    for row in validated["development_report"]["rows"]:
        source_row = source_by_id[row["pair_id"]]
        source, _ = load_raw_default(
            root / source_row["raw_path"], maximum_side
        )
        descriptors.append(
            source_descriptor(
                np.clip(source, 0.0, 1.0), development_config
            )
        )
        parameters.append(row["fitted_parameters"])
        groups.append(source_row["camera_group_id"])
    x = np.stack(descriptors)
    y = np.asarray(parameters, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    alpha = float(config["development_model"]["ridge_alpha"])
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    ensemble = []
    group_array = np.asarray(groups, dtype=object)
    for group in sorted(set(groups)):
        train = group_array != group
        fold_scaler = StandardScaler().fit(x[train])
        fold_model = Ridge(alpha=alpha).fit(
            fold_scaler.transform(x[train]), y[train]
        )
        ensemble.append((fold_scaler, fold_model))
    return {
        "x": x,
        "y": y,
        "scaler": scaler,
        "model": model,
        "ensemble": ensemble,
        "global": np.median(y, axis=0),
    }


def _score_matrix(
    *,
    fresh_x: np.ndarray,
    ridge_parameters: np.ndarray,
    limited_fraction: np.ndarray,
    training: Mapping[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray]:
    scaled_train = training["scaler"].transform(training["x"])
    scaled_fresh = training["scaler"].transform(fresh_x)
    centroid = scaled_train.mean(axis=0)
    centroid_l2 = np.linalg.norm(
        scaled_fresh - centroid[None, :], axis=1
    ) / np.sqrt(scaled_fresh.shape[1])
    nearest_l2 = np.min(
        np.linalg.norm(
            scaled_fresh[:, None, :] - scaled_train[None, :, :],
            axis=2,
        ),
        axis=1,
    ) / np.sqrt(scaled_fresh.shape[1])
    ensemble_predictions = []
    for scaler, model in training["ensemble"]:
        ensemble_predictions.append(
            np.clip(
                model.predict(scaler.transform(fresh_x)),
                lower,
                upper,
            )
        )
    ensemble_array = np.stack(ensemble_predictions)
    span = np.maximum(upper - lower, 1e-12)
    disagreement = np.linalg.norm(
        np.std(ensemble_array / span[None, None, :], axis=0),
        axis=1,
    )
    global_distance = np.linalg.norm(
        (ridge_parameters - training["global"][None, :]) / span[None, :],
        axis=1,
    )
    distance_to_bound = np.minimum(
        (ridge_parameters - lower[None, :]) / span[None, :],
        (upper[None, :] - ridge_parameters) / span[None, :],
    )
    bound_proximity = 1.0 - np.clip(
        2.0 * np.min(distance_to_bound, axis=1), 0.0, 1.0
    )
    return {
        "standardized_centroid_l2": centroid_l2,
        "standardized_nearest_neighbor_l2": nearest_l2,
        "leave_one_training_group_parameter_disagreement_l2": disagreement,
        "prediction_to_global_normalized_l2": global_distance,
        "prediction_bound_proximity": bound_proximity,
        "safe_executor_limited_fraction": limited_fraction,
    }


def _select_policy(
    *,
    train: np.ndarray,
    score_values: Mapping[str, np.ndarray],
    ridge_error: np.ndarray,
    global_error: np.ndarray,
    quantiles: list[float],
) -> dict[str, Any] | None:
    candidates = []
    global_p95 = float(np.quantile(global_error[train], 0.95))
    for score_name in sorted(score_values):
        score = score_values[score_name]
        for quantile in quantiles:
            threshold = float(np.quantile(score[train], quantile))
            active = score[train] <= threshold
            routed = np.where(
                active, ridge_error[train], global_error[train]
            )
            if float(np.quantile(routed, 0.95)) > global_p95 + 1e-12:
                continue
            candidates.append(
                (
                    float(np.mean(routed)),
                    -float(np.mean(active)),
                    score_name,
                    float(quantile),
                    threshold,
                )
            )
    if not candidates:
        return None
    mean_error, negative_coverage, score_name, quantile, threshold = min(
        candidates
    )
    return {
        "score": score_name,
        "quantile": quantile,
        "threshold": threshold,
        "training_mean_error": mean_error,
        "training_active_fraction": -negative_coverage,
    }


def run_diagnostic(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    report_rows = validated["failed_report"]["rows"]
    confirmation_config = _load_hashed_json(
        root,
        config["failed_confirmation"]["config"],
        config["failed_confirmation"]["config_sha256"],
    )
    maximum_side = int(confirmation_config["confirmation"]["maximum_side"])
    confirmation_manifest = _load_hashed_json(
        root,
        confirmation_config["confirmation"]["manifest"],
        confirmation_config["confirmation"]["manifest_sha256"],
    )
    confirmation_by_id = {
        row["pair_id"]: row for row in confirmation_manifest["rows"]
    }
    training = _fit_training_state(
        root=root,
        config=config,
        validated=validated,
        maximum_side=maximum_side,
    )
    development_config = validated["development_config"]
    lower = np.asarray(
        development_config["operator"]["lower_bounds"], dtype=np.float64
    )
    upper = np.asarray(
        development_config["operator"]["upper_bounds"], dtype=np.float64
    )
    descriptors = []
    ridge_parameters = []
    limited_fraction = []
    neutral_ridge = []
    neutral_global = []
    groups = []
    for row in report_rows:
        source_path = Path(
            confirmation_by_id[row["pair_id"]]["source_path"]
        )
        source = load_srgb16(source_path, maximum_side)
        descriptors.append(
            source_descriptor(source, development_config)
        )
        ridge_parameters.append(row["ridge"]["parameters"])
        limited_fraction.append(row["ridge"]["limited_fraction"])
        neutral_ridge.append(row["ridge"]["neutral_rmse"])
        neutral_global.append(row["global"]["neutral_rmse"])
        groups.append(row["camera_model"])
    fresh_x = np.stack(descriptors)
    ridge_parameters_array = np.asarray(
        ridge_parameters, dtype=np.float64
    )
    ridge_error = np.asarray(neutral_ridge, dtype=np.float64)
    global_error = np.asarray(neutral_global, dtype=np.float64)
    scores = _score_matrix(
        fresh_x=fresh_x,
        ridge_parameters=ridge_parameters_array,
        limited_fraction=np.asarray(limited_fraction, dtype=np.float64),
        training=training,
        lower=lower,
        upper=upper,
    )
    harm = ridge_error > global_error
    score_auc = {
        name: float(roc_auc_score(harm.astype(int), values))
        for name, values in scores.items()
    }
    best_score = max(
        score_auc, key=lambda name: (score_auc[name], name)
    )
    best_auc = score_auc[best_score]
    evaluation = config["evaluation"]
    rng = np.random.default_rng(int(evaluation["permutation_seed"]))
    permutation_max = []
    for _ in range(int(evaluation["permutation_count"])):
        permuted = rng.permutation(harm)
        permutation_max.append(
            max(
                roc_auc_score(permuted.astype(int), values)
                for values in scores.values()
            )
        )
    permutation_p = float(
        (1 + np.sum(np.asarray(permutation_max) >= best_auc))
        / (1 + len(permutation_max))
    )
    group_array = np.asarray(groups, dtype=object)
    routed_error = global_error.copy()
    rows = report_rows
    active = np.zeros(len(rows), dtype=bool)
    fold_records = []
    unavailable_folds = 0
    for group in sorted(set(groups)):
        test = group_array == group
        train = ~test
        policy = _select_policy(
            train=train,
            score_values=scores,
            ridge_error=ridge_error,
            global_error=global_error,
            quantiles=[
                float(value)
                for value in config["routing"]["threshold_quantiles"]
            ],
        )
        if policy is None:
            unavailable_folds += 1
            fold_records.append(
                {
                    "held_camera_model": group,
                    "test_rows": int(np.sum(test)),
                    "policy": None,
                }
            )
            continue
        fold_active = scores[policy["score"]][test] <= policy["threshold"]
        active[test] = fold_active
        routed_error[test] = np.where(
            fold_active, ridge_error[test], global_error[test]
        )
        fold_records.append(
            {
                "held_camera_model": group,
                "test_rows": int(np.sum(test)),
                "policy": policy,
                "test_active_rows": int(np.sum(fold_active)),
            }
        )
    oracle_error = np.minimum(ridge_error, global_error)
    active_wins = (
        float(np.mean(ridge_error[active] < global_error[active]))
        if np.any(active)
        else 0.0
    )
    active_groups = len(set(group_array[active]))
    observed = {
        "harm_rows": int(np.sum(harm)),
        "oracle_mean_improvement_over_global": float(
            (global_error.mean() - oracle_error.mean())
            / max(global_error.mean(), 1e-12)
        ),
        "best_score": best_score,
        "best_score_harm_auroc": best_auc,
        "best_score_familywise_permutation_p": permutation_p,
        "crossfit_unavailable_folds": unavailable_folds,
        "crossfit_mean_improvement_over_global": float(
            (global_error.mean() - routed_error.mean())
            / max(global_error.mean(), 1e-12)
        ),
        "crossfit_p95_ratio_to_global": float(
            np.quantile(routed_error, 0.95)
            / max(np.quantile(global_error, 0.95), 1e-12)
        ),
        "crossfit_worst_ratio_to_global": float(
            np.max(routed_error) / max(np.max(global_error), 1e-12)
        ),
        "crossfit_ridge_active_fraction": float(np.mean(active)),
        "active_ridge_win_fraction": active_wins,
        "camera_groups_with_active_ridge": active_groups,
    }
    gates = {
        "oracle_value": observed["oracle_mean_improvement_over_global"]
        >= evaluation["minimum_oracle_mean_improvement_over_global"],
        "score_auc": observed["best_score_harm_auroc"]
        >= evaluation["minimum_best_score_harm_auroc"],
        "score_permutation": observed[
            "best_score_familywise_permutation_p"
        ]
        <= evaluation["maximum_best_score_permutation_p"],
        "all_folds_available": unavailable_folds == 0,
        "crossfit_mean": observed[
            "crossfit_mean_improvement_over_global"
        ]
        >= evaluation["minimum_crossfit_mean_improvement_over_global"],
        "crossfit_p95": observed["crossfit_p95_ratio_to_global"]
        <= evaluation["maximum_crossfit_p95_ratio_to_global"],
        "crossfit_worst": observed["crossfit_worst_ratio_to_global"]
        <= evaluation["maximum_crossfit_worst_ratio_to_global"],
        "active_fraction": observed["crossfit_ridge_active_fraction"]
        >= evaluation["minimum_crossfit_ridge_active_fraction"],
        "active_wins": observed["active_ridge_win_fraction"]
        >= evaluation["minimum_active_ridge_win_fraction"],
        "active_groups": observed["camera_groups_with_active_ridge"]
        >= evaluation["minimum_camera_groups_with_active_ridge"],
    }
    score_rows = [
        {
            "pair_id": row["pair_id"],
            "camera_model": row["camera_model"],
            "harm": bool(harm[index]),
            "ridge_error": float(ridge_error[index]),
            "global_error": float(global_error[index]),
            "routed_error": float(routed_error[index]),
            "crossfit_ridge_active": bool(active[index]),
            "scores": {
                name: float(values[index])
                for name, values in sorted(scores.items())
            },
        }
        for index, row in enumerate(rows)
    ]
    stable = {
        "score_auc": score_auc,
        "observed": observed,
        "gates": gates,
        "folds": fold_records,
        "rows": score_rows,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
    }


__all__ = [
    "FiveKTargetFreeFallbackError",
    "run_diagnostic",
    "validate_contract",
]
