"""Factorized explicit-parameter prediction with analytical safe residuals."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_conditional_explicit_operator import (
    _bounded_transports,
    _metrics,
    effective_case_parameters,
)
from src.eval.fivek_pairwise_compatibility import (
    _group_partition,
    _pooled_operator,
    combined_source_descriptor,
    fit_projection,
    project_features,
)
from src.eval.fivek_source_hard_retrieval import (
    _rgb,
    _source_only_threshold,
    _standardized_distances,
)
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


class FiveKFactorizedSafeResidualError(ValueError):
    """Raised when factorized safe-residual evidence drifts."""


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if (
        features.ndim != 2
        or targets.ndim != 2
        or len(features) != len(targets)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(targets))
    ):
        raise FiveKFactorizedSafeResidualError("invalid ridge arrays")
    x_mean = np.mean(features, axis=0)
    x_scale = np.std(features, axis=0)
    x_scale = np.where(x_scale > 1.0e-8, x_scale, 1.0)
    normalized = (features - x_mean) / x_scale
    y_mean = np.mean(targets, axis=0)
    coefficient = np.linalg.solve(
        normalized.T @ normalized
        + float(alpha) * np.eye(normalized.shape[1], dtype=np.float64),
        normalized.T @ (targets - y_mean),
    )
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "coefficient": coefficient,
    }


def fit_factorized_parameter_model(
    x: np.ndarray,
    parameters: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Fit direction and log magnitude separately in scaled parameter space."""

    targets = np.asarray(parameters, dtype=np.float64)
    if (
        targets.ndim != 2
        or targets.shape[1] != 14
        or not np.all(np.isfinite(targets))
    ):
        raise FiveKFactorizedSafeResidualError("invalid parameter targets")
    parameter_scale = np.maximum(np.max(np.abs(targets), axis=0), 1.0e-6)
    scaled = targets / parameter_scale
    magnitude = np.linalg.norm(scaled, axis=1)
    if np.any(magnitude <= 1.0e-8):
        raise FiveKFactorizedSafeResidualError(
            "identity target cannot define a parameter direction"
        )
    direction = scaled / magnitude[:, None]
    log_magnitude = np.log(magnitude)[:, None]
    return {
        "parameter_scale": parameter_scale,
        "minimum_log_magnitude": float(np.min(log_magnitude)),
        "maximum_log_magnitude": float(np.max(log_magnitude)),
        "direction_model": _fit_ridge(x, direction, alpha),
        "magnitude_model": _fit_ridge(x, log_magnitude, alpha),
    }


def _predict_ridge(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    return (
        (features - np.asarray(model["x_mean"], dtype=np.float64))
        / np.asarray(model["x_scale"], dtype=np.float64)
    ) @ np.asarray(model["coefficient"], dtype=np.float64) + np.asarray(
        model["y_mean"], dtype=np.float64
    )


def predict_factorized_parameters(
    model: Mapping[str, Any], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_direction = _predict_ridge(model["direction_model"], x)
    norm = np.linalg.norm(raw_direction, axis=1)
    if np.any(norm <= 1.0e-8) or not np.all(np.isfinite(norm)):
        raise FiveKFactorizedSafeResidualError(
            "predicted direction is degenerate"
        )
    direction = raw_direction / norm[:, None]
    raw_log_magnitude = _predict_ridge(model["magnitude_model"], x)[:, 0]
    log_magnitude = np.clip(
        raw_log_magnitude,
        float(model["minimum_log_magnitude"]),
        float(model["maximum_log_magnitude"]),
    )
    magnitude = np.exp(log_magnitude)
    parameters = (
        direction
        * magnitude[:, None]
        * np.asarray(model["parameter_scale"], dtype=np.float64)
    )
    if not np.all(np.isfinite(parameters)):
        raise FiveKFactorizedSafeResidualError(
            "predicted parameters are non-finite"
        )
    return parameters, magnitude, raw_log_magnitude != log_magnitude


def analytical_safe_residual(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    epsilon: float,
    margin_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the largest source-to-candidate step that stays strictly interior."""

    source_rgb = np.asarray(source, dtype=np.float64)
    candidate_rgb = np.asarray(candidate, dtype=np.float64)
    if (
        source_rgb.shape != candidate_rgb.shape
        or source_rgb.ndim != 2
        or source_rgb.shape[1] != 3
        or not np.all(np.isfinite(source_rgb))
        or not np.all(np.isfinite(candidate_rgb))
    ):
        raise FiveKFactorizedSafeResidualError("invalid residual arrays")
    lower = float(epsilon) * float(margin_multiplier)
    upper = 1.0 - lower
    if not (0.0 < float(epsilon) < lower < upper < 1.0):
        raise FiveKFactorizedSafeResidualError("invalid interior margin")
    delta = candidate_rgb - source_rgb
    scale = np.ones(len(source_rgb), dtype=np.float64)
    source_boundary = np.any(
        (source_rgb <= epsilon) | (source_rgb >= 1.0 - epsilon), axis=1
    )
    positive = delta > 0.0
    negative = delta < 0.0
    limits = np.full_like(delta, np.inf)
    np.divide(
        upper - source_rgb,
        delta,
        out=limits,
        where=positive,
    )
    np.divide(
        lower - source_rgb,
        delta,
        out=limits,
        where=negative,
    )
    interior = ~source_boundary
    if np.any(interior):
        scale[interior] = np.minimum(
            1.0, np.min(limits[interior], axis=1)
        )
        scale[interior] = np.maximum(scale[interior], 0.0)
    output = source_rgb + scale[:, None] * delta
    if not np.all(np.isfinite(output)):
        raise FiveKFactorizedSafeResidualError("non-finite safe residual")
    return output, scale


def _model_identity(
    projection: tuple[np.ndarray, np.ndarray, np.ndarray],
    model: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for array in projection:
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    for name in ("parameter_scale",):
        digest.update(np.asarray(model[name], dtype="<f8").tobytes())
    for name in ("minimum_log_magnitude", "maximum_log_magnitude"):
        digest.update(np.asarray([model[name]], dtype="<f8").tobytes())
    for branch in ("direction_model", "magnitude_model"):
        for name in ("x_mean", "x_scale", "y_mean", "coefficient"):
            digest.update(
                np.asarray(model[branch][name], dtype="<f8").tobytes()
            )
    return digest.hexdigest()


def _safe_apply(
    transport: TriangularLogitTransport,
    source: np.ndarray,
    evaluation: Mapping[str, Any],
    operator_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    return analytical_safe_residual(
        source,
        transport.apply(source),
        epsilon=float(evaluation["boundary_epsilon"]),
        margin_multiplier=float(
            operator_config["residual_interior_margin_multiplier"]
        ),
    )


def _safe_strength_error(
    pooled: TriangularLogitTransport,
    source: np.ndarray,
    target: np.ndarray,
    doses: Sequence[float],
    evaluation: Mapping[str, Any],
    operator_config: Mapping[str, Any],
) -> float:
    values = []
    for dose in doses:
        transport = TriangularLogitTransport(
            pooled.parameters, dose=pooled.dose * float(dose)
        )
        output, _ = _safe_apply(
            transport, source, evaluation, operator_config
        )
        values.append(_rmse(output, target))
    return min(values)


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, epsilon: float
) -> float:
    source_boundary = np.any(
        (source <= epsilon) | (source >= 1.0 - epsilon), axis=-1
    )
    output_boundary = np.any(
        (output <= epsilon) | (output >= 1.0 - epsilon), axis=-1
    )
    return float(np.mean(output_boundary & ~source_boundary))


def _fit_models(
    features: np.ndarray,
    parameters: np.ndarray,
    fit: np.ndarray,
    model_spec: Mapping[str, Any],
) -> tuple[Any, np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    projection = fit_projection(
        features[fit], int(model_spec["pca_components"])
    )
    projected = project_features(features, *projection)
    model = fit_factorized_parameter_model(
        projected[fit], parameters[fit], float(model_spec["ridge_alpha"])
    )
    rng = np.random.default_rng(int(model_spec["shuffled_parameter_seed"]))
    shuffled = fit_factorized_parameter_model(
        projected[fit],
        parameters[fit][rng.permutation(len(fit))],
        float(model_spec["ridge_alpha"]),
    )
    return projection, projected, model, shuffled


def _evaluate_queries(
    *,
    ordered: Sequence[Mapping[str, Any]],
    query: np.ndarray,
    fit: np.ndarray,
    groups: np.ndarray,
    parameters: np.ndarray,
    projection: Any,
    projected: np.ndarray,
    model: Mapping[str, Any],
    shuffled_model: Mapping[str, Any],
    pooled: TriangularLogitTransport,
    oracle_report: Mapping[str, Any],
    nearest_rows: Mapping[str, Any],
    direct_rows: Mapping[str, Any] | None,
    operator_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
    sample_count: int,
    use_live_oracle: bool,
) -> dict[str, Any]:
    distances, _, _ = _standardized_distances(
        projected[fit], projected[query]
    )
    nearest_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(
        projected[fit], groups[fit], float(evaluation["ood_distance_quantile"])
    )
    fallback = nearest_distance > threshold
    raw, magnitudes, amplitude_clipped = predict_factorized_parameters(
        model, projected[query]
    )
    shuffled_raw, _, _ = predict_factorized_parameters(
        shuffled_model, projected[query]
    )
    transports, parameter_rows, clipped = _bounded_transports(
        raw, parameters[fit], operator_config
    )
    shuffled_transports, _, _ = _bounded_transports(
        shuffled_raw, parameters[fit], operator_config
    )
    case_operators = [
        TriangularLogitTransport(
            np.asarray(row["parameters"], dtype=np.float64),
            dose=float(row["dose"]),
        )
        for row in oracle_report["case_bank"]
    ]
    oracle_rows = {
        str(row["pair_id"]): row for row in oracle_report.get("rows", [])
    }
    arrays: dict[str, list[float]] = {
        name: []
        for name in (
            "baseline", "selected", "identity", "strength", "nearest",
            "oracle", "shuffled", "target_style", "global_style",
            "selected_style", "boundary", "residual_mean_scale", "direct",
        )
    }
    output_rows = []
    epsilon = float(evaluation["boundary_epsilon"])
    for position, index in enumerate(query):
        row = ordered[index]
        pair_id = str(row["pair_id"])
        source = _even_samples(_rgb(row["source"]), sample_count)
        target = _even_samples(_rgb(row["target"]), sample_count)
        global_output, _ = _safe_apply(
            pooled, source, evaluation, operator_config
        )
        selected_transport = pooled if fallback[position] else transports[position]
        output, residual_scale = _safe_apply(
            selected_transport, source, evaluation, operator_config
        )
        shuffled_transport = (
            pooled if fallback[position] else shuffled_transports[position]
        )
        shuffled_output, _ = _safe_apply(
            shuffled_transport, source, evaluation, operator_config
        )
        if use_live_oracle:
            oracle_error = min(
                _rmse(operator.apply(source), target)
                for operator in np.asarray(case_operators, dtype=object)[fit]
            )
        else:
            oracle_error = float(oracle_rows[pair_id]["case_oracle_rmse"])
        direct_error = (
            float(direct_rows[pair_id]["conditional_rmse"])
            if direct_rows is not None
            else float("nan")
        )
        values = {
            "baseline": _rmse(global_output, target),
            "selected": _rmse(output, target),
            "identity": _rmse(source, target),
            "strength": _safe_strength_error(
                pooled,
                source,
                target,
                evaluation["strength_doses"],
                evaluation,
                operator_config,
            ),
            "nearest": float(nearest_rows[pair_id]["selected_rmse"]),
            "oracle": oracle_error,
            "shuffled": _rmse(shuffled_output, target),
            "target_style": _rmse(target, source),
            "global_style": _rmse(global_output, source),
            "selected_style": _rmse(output, source),
            "boundary": _new_boundary_fraction(source, output, epsilon),
            "residual_mean_scale": float(np.mean(residual_scale)),
            "direct": direct_error,
        }
        for name, value in values.items():
            arrays[name].append(value)
        output_rows.append(
            {
                "pair_id": pair_id,
                "group": str(groups[index]),
                **parameter_rows[position],
                "predicted_scaled_parameter_magnitude": float(
                    magnitudes[position]
                ),
                "amplitude_clipped": bool(amplitude_clipped[position]),
                "fallback": bool(fallback[position]),
                "distance": float(nearest_distance[position]),
                "threshold": threshold,
                "identity_rmse": values["identity"],
                "global_rmse": values["baseline"],
                "strength_oracle_rmse": values["strength"],
                "factorized_rmse": values["selected"],
                "direct_bq3_rmse": direct_error,
                "nearest_rmse": values["nearest"],
                "oracle_rmse": values["oracle"],
                "new_boundary_fraction": values["boundary"],
                "mean_residual_scale": values["residual_mean_scale"],
            }
        )
    result = _metrics(
        baseline=np.asarray(arrays["baseline"]),
        selected=np.asarray(arrays["selected"]),
        identity=np.asarray(arrays["identity"]),
        strength=np.asarray(arrays["strength"]),
        nearest=np.asarray(arrays["nearest"]),
        oracle=np.asarray(arrays["oracle"]),
        shuffled=np.asarray(arrays["shuffled"]),
        target_style=np.asarray(arrays["target_style"]),
        global_style=np.asarray(arrays["global_style"]),
        selected_style=np.asarray(arrays["selected_style"]),
        groups=groups[query],
        doses=np.asarray([row["dose"] for row in parameter_rows]),
        clipping_fraction=float(np.mean(clipped)),
        maximum_new_boundary=float(np.max(arrays["boundary"])),
        fallback=fallback,
        gates=gates,
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
    )
    selected = np.asarray(arrays["selected"])
    if direct_rows is not None:
        direct = np.asarray(arrays["direct"])
        direct_gain = (
            float(np.mean(direct)) - float(np.mean(selected))
        ) / max(float(np.mean(direct)), 1.0e-12)
        direct_wins = float(np.mean(selected < direct))
        result["metrics"]["mean_improvement_over_direct_bq3"] = direct_gain
        result["metrics"]["win_fraction_over_direct_bq3"] = direct_wins
        result["gates"]["direct_mean"] = direct_gain >= float(
            gates["minimum_mean_improvement_over_direct_bq3"]
        )
        result["gates"]["direct_wins"] = direct_wins >= float(
            gates["minimum_win_fraction_over_direct_bq3"]
        )
    residual_median = float(np.median(arrays["residual_mean_scale"]))
    result["metrics"]["median_image_mean_residual_scale"] = residual_median
    result["metrics"]["amplitude_clipping_fraction"] = float(
        np.mean(amplitude_clipped)
    )
    result["gates"]["residual_scale"] = residual_median >= float(
        gates["minimum_median_image_mean_residual_scale"]
    )
    result["automatic_pass"] = all(result["gates"].values())
    result.update(
        {
            "model_sha256": _model_identity(projection, model),
            "rows": output_rows,
        }
    )
    return result


def evaluate_development(
    *,
    rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    nearest_report: Mapping[str, Any],
    direct_report: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    if ids != [str(row["pair_id"]) for row in oracle_report["case_bank"]]:
        raise FiveKFactorizedSafeResidualError("case identity drift")
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    features = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in ordered]
    )
    parameters = effective_case_parameters(oracle_report)
    fit, query = _group_partition(
        groups,
        int(model_spec["group_bucket_modulus"]),
        int(model_spec["validation_bucket"]),
    )
    projection, projected, model, shuffled = _fit_models(
        features, parameters, fit, model_spec
    )
    result = _evaluate_queries(
        ordered=ordered,
        query=query,
        fit=fit,
        groups=groups,
        parameters=parameters,
        projection=projection,
        projected=projected,
        model=model,
        shuffled_model=shuffled,
        pooled=_pooled_operator(ordered, fit, operator_config),
        oracle_report=oracle_report,
        nearest_rows={str(row["pair_id"]): row for row in nearest_report["rows"]},
        direct_rows={str(row["pair_id"]): row for row in direct_report["rows"]},
        operator_config=operator_config,
        evaluation=evaluation,
        gates=gates,
        sample_count=int(evaluation["development_samples_per_image"]),
        use_live_oracle=True,
    )
    result.update(
        {
            "fit_rows": len(fit),
            "validation_rows": len(query),
            "fit_groups": sorted(set(map(str, groups[fit]))),
            "validation_groups": sorted(set(map(str, groups[query]))),
        }
    )
    return result


def evaluate_confirmation(
    *,
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    nearest_report: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    development = sorted(
        development_rows, key=lambda row: str(row["pair_id"])
    )
    confirmation = sorted(
        confirmation_rows, key=lambda row: str(row["pair_id"])
    )
    ordered = development + confirmation
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    features = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in ordered]
    )
    parameters = effective_case_parameters(oracle_report)
    fit = np.arange(len(development), dtype=np.int64)
    query = np.arange(len(development), len(ordered), dtype=np.int64)
    projection, projected, model, shuffled = _fit_models(
        features, parameters, fit, model_spec
    )
    pooled = TriangularLogitTransport(
        np.asarray(oracle_report["pooled_operator"]["parameters"], dtype=np.float64),
        dose=float(oracle_report["pooled_operator"]["dose"]),
    )
    return _evaluate_queries(
        ordered=ordered,
        query=query,
        fit=fit,
        groups=groups,
        parameters=parameters,
        projection=projection,
        projected=projected,
        model=model,
        shuffled_model=shuffled,
        pooled=pooled,
        oracle_report=oracle_report,
        nearest_rows={str(row["pair_id"]): row for row in nearest_report["rows"]},
        direct_rows=None,
        operator_config=operator_config,
        evaluation=evaluation,
        gates=gates,
        sample_count=int(evaluation["confirmation_samples_per_image"]),
        use_live_oracle=False,
    )


__all__ = [
    "FiveKFactorizedSafeResidualError",
    "analytical_safe_residual",
    "evaluate_confirmation",
    "evaluate_development",
    "fit_factorized_parameter_model",
    "predict_factorized_parameters",
]
