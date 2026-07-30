"""Group-held-out development of explicit monotone photographic curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.eval.boundary_safe_neutral_base import (
    apply_boundary_safe_residual,
    validate_contract as validate_safe_contract,
)
from src.eval.fivek_hard_case_medoid_development import (
    _load_ay0_population,
    _load_fresh_population,
)
from src.eval.fivek_neutral_base_fixed_ao6_ablation import (
    build_fixed_ao6_renderer,
)
from src.eval.fivek_unseen_content_confirmation import (
    _median_delta_e76,
    _new_boundary_fraction,
    _rmse,
    _summary,
)


class FiveKMonotoneCurveError(ValueError):
    """Raised when curve evidence or the frozen representation drifts."""


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
    root: Path, item: Mapping[str, Any], key: str
) -> dict[str, Any]:
    path = root / str(item[key])
    if not path.is_file() or _sha256(path) != str(
        item[f"{key}_sha256"]
    ).lower():
        raise FiveKMonotoneCurveError(f"evidence drift: {item[key]}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKMonotoneCurveError("contract is not frozen")
    operator = config["operator"]
    knots = np.asarray(operator["knot_inputs"], dtype=np.float64)
    if (
        config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or operator.get("hard_output_clipping_allowed")
        or operator.get("channel_independent_residual_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or len(knots) != 7
        or not np.array_equal(knots[[0, -1]], [0.0, 1.0])
        or not np.all(np.diff(knots) > 0.0)
        or operator.get("parameter_count") != 15
    ):
        raise FiveKMonotoneCurveError("curve representation drift")
    parents = config["parents"]
    ay0_config = _load_hashed_json(root, parents["ay0"], "config")
    ay0_report = _load_hashed_json(root, parents["ay0"], "report")
    safe_config = _load_hashed_json(root, parents["ay3"], "config")
    safe_decision = _load_hashed_json(root, parents["ay3"], "decision")
    fresh_manifest = _load_hashed_json(
        root, parents["fresh_population"], "manifest"
    )
    fresh_report = _load_hashed_json(
        root, parents["fresh_population"], "report"
    )
    fresh_decision = _load_hashed_json(
        root, parents["fresh_population"], "decision"
    )
    hard_config = _load_hashed_json(
        root, parents["closed_hard_cases"], "config"
    )
    hard_decision = _load_hashed_json(
        root, parents["closed_hard_cases"], "decision"
    )
    safe_validated = validate_safe_contract(root, safe_config)
    if (
        ay0_report.get("automatic_pass") is not True
        or safe_decision.get("status")
        != "development_pass_fresh_confirmation_required"
        or fresh_report.get("automatic_pass") is not True
        or fresh_decision.get("status") != "pass_fresh_confirmation_ready"
        or len(fresh_manifest.get("rows", [])) != 63
        or hard_decision.get("status")
        != parents["closed_hard_cases"]["required_status"]
        or hard_config.get("experiment_id")
        != "u5.r2ay5-hard-case-medoid-development-v1"
    ):
        raise FiveKMonotoneCurveError("parent eligibility drift")
    return {
        "ay0_config": ay0_config,
        "ay0_report": ay0_report,
        "safe_config": safe_config,
        "safe_validated": safe_validated,
        "fresh_manifest": fresh_manifest,
    }


def project_curve_parameters(
    parameters: np.ndarray,
    *,
    knots: np.ndarray,
    endpoint_weight: float,
) -> np.ndarray:
    """Project 15 ordinates to three endpoint-exact monotone curves."""

    values = np.asarray(parameters, dtype=np.float64)
    single = values.ndim == 1
    values = values.reshape(-1, 3, len(knots) - 2)
    projected = np.empty_like(values)
    weights = np.ones(len(knots), dtype=np.float64)
    weights[[0, -1]] = endpoint_weight
    for row in range(len(values)):
        for channel in range(3):
            full = np.concatenate(
                ([0.0], values[row, channel], [1.0])
            )
            fitted = IsotonicRegression(
                increasing=True,
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            ).fit_transform(knots, full, sample_weight=weights)
            fitted[0] = 0.0
            fitted[-1] = 1.0
            projected[row, channel] = fitted[1:-1]
    flattened = projected.reshape(len(values), -1)
    return flattened[0] if single else flattened


def fit_curve_parameters(
    source: np.ndarray,
    target: np.ndarray,
    *,
    knots: np.ndarray,
    sample_stride: int,
    endpoint_weight: float,
) -> np.ndarray:
    """Fit three monotone marginal transfer curves and sample fixed knots."""

    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 3
        or source_array.shape[2] != 3
        or sample_stride < 1
    ):
        raise FiveKMonotoneCurveError("invalid curve fit peers")
    weights = np.ones(
        source_array[::sample_stride, ::sample_stride, 0].size + 2,
        dtype=np.float64,
    )
    weights[[-2, -1]] = endpoint_weight
    channels = []
    for channel in range(3):
        x = source_array[
            ::sample_stride, ::sample_stride, channel
        ].reshape(-1)
        y = target_array[
            ::sample_stride, ::sample_stride, channel
        ].reshape(-1)
        model = IsotonicRegression(
            increasing=True,
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit(
            np.concatenate((x, [0.0, 1.0])),
            np.concatenate((y, [0.0, 1.0])),
            sample_weight=weights,
        )
        curve = model.predict(knots)
        curve[0] = 0.0
        curve[-1] = 1.0
        channels.append(curve[1:-1])
    return project_curve_parameters(
        np.concatenate(channels),
        knots=knots,
        endpoint_weight=endpoint_weight,
    )


def apply_curve_operator(
    source: np.ndarray, parameters: np.ndarray, *, knots: np.ndarray
) -> np.ndarray:
    """Apply endpoint-exact piecewise-linear channel curves."""

    source_array = np.asarray(source, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64).reshape(
        3, len(knots) - 2
    )
    output = np.empty_like(source_array)
    for channel in range(3):
        ordinates = np.concatenate(
            ([0.0], values[channel], [1.0])
        )
        if np.any(np.diff(ordinates) < -1e-12):
            raise FiveKMonotoneCurveError("curve is not monotone")
        output[..., channel] = np.interp(
            source_array[..., channel], knots, ordinates
        )
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise FiveKMonotoneCurveError("curve escaped the RGB cube")
    return output


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    knots: np.ndarray,
    endpoint_weight: float,
) -> float:
    unique_groups = len(set(groups.tolist()))
    splits = min(4, unique_groups)
    if splits < 2:
        raise FiveKMonotoneCurveError("insufficient alpha groups")
    scores = []
    for alpha in alphas:
        fold_errors = []
        for train, valid in GroupKFold(n_splits=splits).split(
            x, groups=groups
        ):
            scaler = StandardScaler().fit(x[train])
            predicted = Ridge(alpha=alpha).fit(
                scaler.transform(x[train]), y[train]
            ).predict(scaler.transform(x[valid]))
            predicted = project_curve_parameters(
                predicted,
                knots=knots,
                endpoint_weight=endpoint_weight,
            )
            fold_errors.extend(
                np.mean((predicted - y[valid]) ** 2, axis=1).tolist()
            )
        scores.append((float(np.mean(fold_errors)), alpha))
    return min(scores)[1]


def _predict_curves(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    *,
    alphas: list[float],
    knots: np.ndarray,
    endpoint_weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = _select_alpha(
        train_x,
        train_y,
        train_groups,
        alphas,
        knots=knots,
        endpoint_weight=endpoint_weight,
    )
    scaler = StandardScaler().fit(train_x)
    ridge = Ridge(alpha=alpha).fit(
        scaler.transform(train_x), train_y
    )
    predicted = project_curve_parameters(
        ridge.predict(scaler.transform(test_x)),
        knots=knots,
        endpoint_weight=endpoint_weight,
    )
    global_parameters = project_curve_parameters(
        np.median(train_y, axis=0),
        knots=knots,
        endpoint_weight=endpoint_weight,
    )
    return (
        predicted,
        np.repeat(global_parameters[None, :], len(test_x), axis=0),
        alpha,
    )


def _prepare_parameters(
    ay0: Mapping[str, Any],
    fresh: Mapping[str, Any],
    *,
    knots: np.ndarray,
    stride: int,
    endpoint_weight: float,
    alphas: list[float],
) -> dict[str, Any]:
    for population in (ay0, fresh):
        for row in population["rows"]:
            row["curve_fitted"] = fit_curve_parameters(
                row["source"],
                row["target"],
                knots=knots,
                sample_stride=stride,
                endpoint_weight=endpoint_weight,
            )
    train_x = np.stack([row["descriptor"] for row in ay0["rows"]])
    train_y = np.stack([row["curve_fitted"] for row in ay0["rows"]])
    train_groups = np.asarray(
        [row["group"] for row in ay0["rows"]], dtype=object
    )
    outer_predictions = np.empty_like(train_y)
    outer_global = np.empty_like(train_y)
    outer_alpha = []
    for train, test in GroupKFold(n_splits=5).split(
        train_x, groups=train_groups
    ):
        predicted, global_parameters, alpha = _predict_curves(
            train_x[train],
            train_y[train],
            train_groups[train],
            train_x[test],
            alphas=alphas,
            knots=knots,
            endpoint_weight=endpoint_weight,
        )
        outer_predictions[test] = predicted
        outer_global[test] = global_parameters
        outer_alpha.append(
            {
                "held_groups": sorted(
                    set(train_groups[test].tolist())
                ),
                "alpha": alpha,
            }
        )
    fresh_x = np.stack(
        [row["descriptor"] for row in fresh["rows"]]
    )
    fresh_ridge, fresh_global, full_alpha = _predict_curves(
        train_x,
        train_y,
        train_groups,
        fresh_x,
        alphas=alphas,
        knots=knots,
        endpoint_weight=endpoint_weight,
    )
    return {
        ay0["name"]: {
            "ridge": outer_predictions,
            "global": outer_global,
            "alphas": outer_alpha,
        },
        fresh["name"]: {
            "ridge": fresh_ridge,
            "global": fresh_global,
            "alphas": [{"full_training_alpha": full_alpha}],
        },
    }


def _evaluate_population(
    population: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    knots: np.ndarray,
    epsilon: float,
    endpoint_weight: float,
    renderer: Any,
) -> dict[str, Any]:
    methods = ("identity", "global", "ridge", "oracle")
    identity = project_curve_parameters(
        np.tile(knots[1:-1], 3),
        knots=knots,
        endpoint_weight=endpoint_weight,
    )
    neutral_errors = {method: [] for method in methods}
    look_errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    limited = {method: [] for method in methods}
    rows = []
    for index, row in enumerate(population["rows"]):
        parameter_map = {
            "identity": identity,
            "global": predictions["global"][index],
            "ridge": predictions["ridge"][index],
            "oracle": row["curve_fitted"],
        }
        target_look, _ = renderer(row["target"])
        record: dict[str, Any] = {
            "pair_id": row["pair_id"],
            "group": row["group"],
            "target_style_delta_e76": _median_delta_e76(
                row["target"], target_look
            ),
        }
        for method in methods:
            candidate = apply_curve_operator(
                row["source"], parameter_map[method], knots=knots
            )
            neutral, scale = apply_boundary_safe_residual(
                row["source"],
                candidate,
                boundary_epsilon=epsilon,
            )
            look, _ = renderer(neutral)
            neutral_error = _rmse(neutral, row["target"])
            look_error = _rmse(look, target_look)
            style = _median_delta_e76(neutral, look)
            boundary = max(
                _new_boundary_fraction(row["source"], neutral, epsilon),
                _new_boundary_fraction(neutral, look, epsilon),
            )
            limited_fraction = float(np.mean(scale < 1.0 - 1e-12))
            neutral_errors[method].append(neutral_error)
            look_errors[method].append(look_error)
            styles[method].append(style)
            boundaries[method].append(boundary)
            limited[method].append(limited_fraction)
            record[method] = {
                "parameters": parameter_map[method].tolist(),
                "neutral_rmse": neutral_error,
                "look_rmse": look_error,
                "style_delta_e76": style,
                "limited_fraction": limited_fraction,
                "new_boundary_fraction": boundary,
            }
        rows.append(record)
    metrics = {}
    global_error = np.asarray(neutral_errors["global"])
    identity_error = np.asarray(neutral_errors["identity"])
    for method in methods:
        error = np.asarray(neutral_errors[method])
        metrics[method] = {
            "neutral": _summary(neutral_errors[method]),
            "look": _summary(look_errors[method]),
            "mean_improvement_over_global": float(
                (global_error.mean() - error.mean())
                / max(global_error.mean(), 1e-12)
            ),
            "win_fraction_over_global": float(
                np.mean(error < global_error)
            ),
            "p95_ratio_to_global": float(
                np.quantile(error, 0.95)
                / max(np.quantile(global_error, 0.95), 1e-12)
            ),
            "worst_ratio_to_global": float(
                np.max(error) / max(np.max(global_error), 1e-12)
            ),
            "mean_improvement_over_identity": float(
                (identity_error.mean() - error.mean())
                / max(identity_error.mean(), 1e-12)
            ),
            "style_ratio_to_global": float(
                np.median(styles[method])
                / max(np.median(styles["global"]), 1e-12)
            ),
            "median_style_delta_e76": float(
                np.median(styles[method])
            ),
            "median_limited_fraction": float(
                np.median(limited[method])
            ),
            "p95_limited_fraction": float(
                np.quantile(limited[method], 0.95)
            ),
            "maximum_new_boundary_fraction": float(
                np.max(boundaries[method])
            ),
        }
    return {
        "name": population["name"],
        "row_count": len(rows),
        "group_count": len(
            set(row["group"] for row in population["rows"])
        ),
        "selected_alphas": predictions["alphas"],
        "metrics": metrics,
        "rows": rows,
    }


def run_development(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    ay0 = _load_ay0_population(
        root, validated["ay0_config"], validated["ay0_report"]
    )
    fresh = _load_fresh_population(
        root, validated["ay0_config"], validated["fresh_manifest"]
    )
    operator = config["operator"]
    knots = np.asarray(operator["knot_inputs"], dtype=np.float64)
    endpoint_weight = float(operator["endpoint_anchor_weight"])
    predictions = _prepare_parameters(
        ay0,
        fresh,
        knots=knots,
        stride=int(operator["fit_sample_stride"]),
        endpoint_weight=endpoint_weight,
        alphas=[
            float(value) for value in config["prediction"]["ridge_alphas"]
        ],
    )
    renderer = build_fixed_ao6_renderer(
        validated["safe_validated"]["fixed_config"],
        validated["safe_validated"]["fixed_validated"],
    )
    epsilon = float(config["parents"]["ay3"]["boundary_epsilon"])
    populations = {
        ay0["name"]: _evaluate_population(
            ay0,
            predictions[ay0["name"]],
            knots=knots,
            epsilon=epsilon,
            endpoint_weight=endpoint_weight,
            renderer=renderer,
        ),
        fresh["name"]: _evaluate_population(
            fresh,
            predictions[fresh["name"]],
            knots=knots,
            epsilon=epsilon,
            endpoint_weight=endpoint_weight,
            renderer=renderer,
        ),
    }
    thresholds = config["evaluation"]
    gates = {}
    for name, population in populations.items():
        ridge = population["metrics"]["ridge"]
        oracle = population["metrics"]["oracle"]
        gates[name] = {
            "oracle_capacity": oracle["mean_improvement_over_identity"]
            >= thresholds[
                "minimum_oracle_mean_improvement_over_identity_each_population"
            ],
            "ridge_mean": ridge["mean_improvement_over_global"]
            >= thresholds[
                "minimum_ridge_mean_improvement_over_global_each_population"
            ],
            "ridge_wins": ridge["win_fraction_over_global"]
            >= thresholds[
                "minimum_ridge_win_fraction_over_global_each_population"
            ],
            "ridge_p95": ridge["p95_ratio_to_global"]
            <= thresholds[
                "maximum_ridge_p95_ratio_to_global_each_population"
            ],
            "ridge_worst": ridge["worst_ratio_to_global"]
            <= thresholds[
                "maximum_ridge_worst_ratio_to_global_each_population"
            ],
            "ridge_style": ridge["style_ratio_to_global"]
            >= thresholds[
                "minimum_ridge_ao6_style_ratio_to_global_each_population"
            ],
            "median_limited": ridge["median_limited_fraction"]
            <= thresholds[
                "maximum_ridge_median_limited_fraction_each_population"
            ],
            "p95_limited": ridge["p95_limited_fraction"]
            <= thresholds[
                "maximum_ridge_p95_limited_fraction_each_population"
            ],
            "boundary": max(
                method["maximum_new_boundary_fraction"]
                for method in population["metrics"].values()
            )
            <= thresholds["maximum_new_boundary_fraction"],
        }
    stable = {"populations": populations, "gates": gates}
    automatic_pass = all(
        all(population_gates.values())
        for population_gates in gates.values()
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "automatic_pass": automatic_pass,
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
    "FiveKMonotoneCurveError",
    "apply_curve_operator",
    "fit_curve_parameters",
    "project_curve_parameters",
    "run_development",
    "validate_contract",
]
