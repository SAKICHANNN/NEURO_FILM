"""Source-conditioned bounded explicit operators for the large FiveK control."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import (
    _group_partition,
    _model_identity,
    _pooled_operator,
    combined_source_descriptor,
    fit_projection,
    project_features,
)
from src.eval.fivek_source_hard_retrieval import (
    _group_bootstrap,
    _rgb,
    _source_only_threshold,
    _standardized_distances,
)
from src.roll2film.triangular_logit_transport import (
    TriangularLogitTransport,
    select_safe_transport,
)


class FiveKConditionalExplicitOperatorError(ValueError):
    """Raised when conditional explicit-operator evidence drifts."""


def effective_case_parameters(oracle_report: Mapping[str, Any]) -> np.ndarray:
    rows = oracle_report["case_bank"]
    values = np.stack(
        [
            np.asarray(row["parameters"], dtype=np.float64)
            * float(row["dose"])
            for row in rows
        ]
    )
    if values.ndim != 2 or values.shape[1] != 14 or not np.all(
        np.isfinite(values)
    ):
        raise FiveKConditionalExplicitOperatorError(
            "invalid effective case parameters"
        )
    return values


def fit_parameter_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if (
        features.ndim != 2
        or targets.ndim != 2
        or len(features) != len(targets)
        or targets.shape[1] != 14
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(targets))
    ):
        raise FiveKConditionalExplicitOperatorError(
            "invalid parameter regression arrays"
        )
    x_mean = np.mean(features, axis=0)
    x_scale = np.std(features, axis=0)
    x_scale = np.where(x_scale > 1.0e-8, x_scale, 1.0)
    normalized = (features - x_mean) / x_scale
    y_mean = np.mean(targets, axis=0)
    gram = normalized.T @ normalized
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
        normalized.T @ (targets - y_mean),
    )
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "coefficient": coefficient,
    }


def predict_parameters(
    model: Mapping[str, Any], features: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    result = (
        (x - np.asarray(model["x_mean"]))
        / np.asarray(model["x_scale"])
    ) @ np.asarray(model["coefficient"]) + np.asarray(model["y_mean"])
    if result.ndim != 2 or result.shape[1] != 14 or not np.all(
        np.isfinite(result)
    ):
        raise FiveKConditionalExplicitOperatorError(
            "non-finite predicted parameters"
        )
    return result


def _parameter_model_identity(
    projection: tuple[np.ndarray, np.ndarray, np.ndarray],
    model: Mapping[str, Any],
) -> str:
    compatible = {
        "mean": model["x_mean"],
        "scale": model["x_scale"],
        "coefficient": np.asarray(model["coefficient"]).ravel(),
        "intercept": 0.0,
    }
    digest = hashlib.sha256()
    digest.update(_model_identity(projection, compatible).encode("ascii"))
    digest.update(np.asarray(model["y_mean"], dtype="<f8").tobytes())
    return digest.hexdigest()


def _bounded_transports(
    raw_parameters: np.ndarray,
    training_parameters: np.ndarray,
    operator_config: Mapping[str, Any],
) -> tuple[list[TriangularLogitTransport], list[dict[str, Any]], np.ndarray]:
    lower = np.maximum(
        np.min(training_parameters, axis=0),
        np.asarray(operator_config["lower_bounds"], dtype=np.float64),
    )
    upper = np.minimum(
        np.max(training_parameters, axis=0),
        np.asarray(operator_config["upper_bounds"], dtype=np.float64),
    )
    bounded = np.clip(raw_parameters, lower, upper)
    clipped = ~np.isclose(raw_parameters, bounded, rtol=0.0, atol=1.0e-12)
    transports = []
    rows = []
    for parameters in bounded:
        transport, diagnostics = select_safe_transport(
            parameters=parameters,
            grid_size=int(operator_config["safe_dose_grid_size"]),
            finite_difference=float(
                operator_config["safe_dose_finite_difference"]
            ),
            minimum_jacobian_determinant=float(
                operator_config["minimum_jacobian_determinant"]
            ),
            maximum_jacobian_condition=float(
                operator_config["maximum_jacobian_condition"]
            ),
            bisection_iterations=int(
                operator_config["safe_dose_bisection_iterations"]
            ),
        )
        transports.append(transport)
        rows.append(
            {
                "parameters": transport.parameters.tolist(),
                "effective_parameters": transport.effective_parameters.tolist(),
                "dose": float(transport.dose),
                "minimum_jacobian_determinant": float(
                    diagnostics["minimum_determinant"]
                ),
                "maximum_jacobian_condition": float(
                    diagnostics["maximum_condition"]
                ),
            }
        )
    return transports, rows, clipped


def _strength_error(
    pooled: TriangularLogitTransport,
    source: np.ndarray,
    target: np.ndarray,
    doses: Sequence[float],
) -> float:
    return min(
        _rmse(
            TriangularLogitTransport(
                pooled.parameters, dose=pooled.dose * float(dose)
            ).apply(source),
            target,
        )
        for dose in doses
    )


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


def _metrics(
    *,
    baseline: np.ndarray,
    selected: np.ndarray,
    identity: np.ndarray,
    strength: np.ndarray,
    nearest: np.ndarray,
    oracle: np.ndarray,
    shuffled: np.ndarray,
    target_style: np.ndarray,
    global_style: np.ndarray,
    selected_style: np.ndarray,
    groups: np.ndarray,
    doses: np.ndarray,
    clipping_fraction: float,
    maximum_new_boundary: float,
    fallback: np.ndarray,
    gates: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    bootstrap = _group_bootstrap(
        baseline,
        selected,
        groups,
        seed=bootstrap_seed,
        repetitions=bootstrap_repetitions,
    )
    selected_mean = float(np.mean(selected))

    def improvement(other: np.ndarray) -> float:
        mean = float(np.mean(other))
        return (mean - selected_mean) / max(mean, 1.0e-12)

    metrics = {
        "mean_improvement_over_identity": improvement(identity),
        "mean_improvement_over_global": improvement(baseline),
        "win_fraction_over_global": float(np.mean(selected < baseline)),
        "p95_ratio_to_global": float(np.quantile(selected, 0.95))
        / max(float(np.quantile(baseline, 0.95)), 1.0e-12),
        "worst_ratio_to_global": float(np.max(selected))
        / max(float(np.max(baseline)), 1.0e-12),
        "oracle_gap_closure": (
            float(np.mean(baseline)) - selected_mean
        )
        / max(float(np.mean(baseline)) - float(np.mean(oracle)), 1.0e-12),
        "mean_improvement_over_strength_oracle": improvement(strength),
        "win_fraction_over_strength_oracle": float(
            np.mean(selected < strength)
        ),
        "mean_improvement_over_nearest": improvement(nearest),
        "win_fraction_over_nearest": float(np.mean(selected < nearest)),
        "mean_improvement_over_shuffled_parameters": improvement(shuffled),
        "group_bootstrap_improvement_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "median_style_retention_ratio": float(
            np.median(selected_style / np.maximum(target_style, 1.0e-12))
        ),
        "mean_style_magnitude_ratio_to_global": float(
            np.mean(selected_style) / max(float(np.mean(global_style)), 1.0e-12)
        ),
        "minimum_safe_dose": float(np.min(doses)),
        "median_safe_dose": float(np.median(doses)),
        "parameter_clipping_fraction": float(clipping_fraction),
        "maximum_new_boundary_fraction": float(maximum_new_boundary),
        "fallback_fraction": float(np.mean(fallback)),
    }
    checks = {
        "identity": metrics["mean_improvement_over_identity"]
        >= gates["minimum_mean_improvement_over_identity"],
        "mean": metrics["mean_improvement_over_global"]
        >= gates["minimum_mean_improvement_over_global"],
        "wins": metrics["win_fraction_over_global"]
        >= gates["minimum_win_fraction_over_global"],
        "p95": metrics["p95_ratio_to_global"]
        <= gates["maximum_p95_ratio_to_global"],
        "worst": metrics["worst_ratio_to_global"]
        <= gates["maximum_worst_ratio_to_global"],
        "oracle_gap": metrics["oracle_gap_closure"]
        >= gates["minimum_oracle_gap_closure"],
        "strength_mean": metrics["mean_improvement_over_strength_oracle"]
        >= gates["minimum_mean_improvement_over_strength_oracle"],
        "strength_wins": metrics["win_fraction_over_strength_oracle"]
        >= gates["minimum_win_fraction_over_strength_oracle"],
        "nearest_mean": metrics["mean_improvement_over_nearest"]
        >= gates["minimum_mean_improvement_over_nearest"],
        "nearest_wins": metrics["win_fraction_over_nearest"]
        >= gates["minimum_win_fraction_over_nearest"],
        "shuffle": metrics["mean_improvement_over_shuffled_parameters"]
        >= gates["minimum_mean_improvement_over_shuffled_parameters"],
        "bootstrap": metrics["group_bootstrap_improvement_ci95"][0]
        > gates["minimum_bootstrap_lower_improvement"],
        "style_target": metrics["median_style_retention_ratio"]
        >= gates["minimum_median_style_retention_ratio"],
        "style_global": metrics["mean_style_magnitude_ratio_to_global"]
        >= gates["minimum_mean_style_magnitude_ratio_to_global"],
        "safe_dose": metrics["minimum_safe_dose"]
        >= gates["minimum_safe_dose"],
        "parameter_clipping": metrics["parameter_clipping_fraction"]
        <= gates["maximum_parameter_clipping_fraction"],
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "fallback": metrics["fallback_fraction"]
        <= gates["maximum_fallback_fraction"],
    }
    return {"metrics": metrics, "gates": checks, "automatic_pass": all(checks.values())}


def _fit_models(
    *,
    features: np.ndarray,
    parameters: np.ndarray,
    fit: np.ndarray,
    model_spec: Mapping[str, Any],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    projection = fit_projection(
        features[fit], int(model_spec["pca_components"])
    )
    projected = project_features(features, *projection)
    model = fit_parameter_ridge(
        projected[fit], parameters[fit], float(model_spec["ridge_alpha"])
    )
    rng = np.random.default_rng(int(model_spec["shuffled_parameter_seed"]))
    shuffled_model = fit_parameter_ridge(
        projected[fit],
        parameters[fit][rng.permutation(len(fit))],
        float(model_spec["ridge_alpha"]),
    )
    return projection, projected, model, shuffled_model


def evaluate_development(
    *,
    rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    nearest_report: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    report_ids = [str(row["pair_id"]) for row in oracle_report["case_bank"]]
    if ids != report_ids:
        raise FiveKConditionalExplicitOperatorError("case identity drift")
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    features = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in ordered]
    )
    parameters = effective_case_parameters(oracle_report)
    fit, validation = _group_partition(
        groups,
        int(model_spec["group_bucket_modulus"]),
        int(model_spec["validation_bucket"]),
    )
    projection, projected, model, shuffled_model = _fit_models(
        features=features, parameters=parameters, fit=fit, model_spec=model_spec
    )
    distances, _, _ = _standardized_distances(
        projected[fit], projected[validation]
    )
    nearest_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(
        projected[fit],
        groups[fit],
        float(evaluation["ood_distance_quantile"]),
    )
    fallback = nearest_distance > threshold
    raw = predict_parameters(model, projected[validation])
    shuffled_raw = predict_parameters(shuffled_model, projected[validation])
    transports, parameter_rows, clipped = _bounded_transports(
        raw, parameters[fit], operator_config
    )
    shuffled_transports, _, _ = _bounded_transports(
        shuffled_raw, parameters[fit], operator_config
    )
    pooled = _pooled_operator(ordered, fit, operator_config)
    nearest_rows = {str(row["pair_id"]): row for row in nearest_report["rows"]}
    operators = [
        TriangularLogitTransport(
            np.asarray(row["parameters"], dtype=np.float64),
            dose=float(row["dose"]),
        )
        for row in oracle_report["case_bank"]
    ]
    samples = int(evaluation["development_samples_per_image"])
    epsilon = float(evaluation["boundary_epsilon"])
    arrays: dict[str, list[float]] = {
        name: []
        for name in (
            "baseline", "selected", "identity", "strength", "nearest",
            "oracle", "shuffled", "target_style", "global_style",
            "selected_style", "boundary",
        )
    }
    output_rows = []
    for position, index in enumerate(validation):
        source = _even_samples(_rgb(ordered[index]["source"]), samples)
        target = _even_samples(_rgb(ordered[index]["target"]), samples)
        global_output = pooled.apply(source)
        output = (
            global_output
            if fallback[position]
            else transports[position].apply(source)
        )
        shuffled_output = (
            global_output
            if fallback[position]
            else shuffled_transports[position].apply(source)
        )
        case_errors = np.asarray(
            [_rmse(operators[case].apply(source), target) for case in fit]
        )
        values = {
            "baseline": _rmse(global_output, target),
            "selected": _rmse(output, target),
            "identity": _rmse(source, target),
            "strength": _strength_error(
                pooled, source, target, evaluation["strength_doses"]
            ),
            "nearest": float(nearest_rows[ids[index]]["selected_rmse"]),
            "oracle": float(np.min(case_errors)),
            "shuffled": _rmse(shuffled_output, target),
            "target_style": _rmse(target, source),
            "global_style": _rmse(global_output, source),
            "selected_style": _rmse(output, source),
            "boundary": _new_boundary_fraction(source, output, epsilon),
        }
        for name, value in values.items():
            arrays[name].append(value)
        output_rows.append(
            {
                "pair_id": ids[index],
                "group": str(groups[index]),
                **parameter_rows[position],
                "fallback": bool(fallback[position]),
                "distance": float(nearest_distance[position]),
                "threshold": threshold,
                "identity_rmse": values["identity"],
                "global_rmse": values["baseline"],
                "strength_oracle_rmse": values["strength"],
                "conditional_rmse": values["selected"],
                "nearest_rmse": values["nearest"],
                "oracle_rmse": values["oracle"],
                "new_boundary_fraction": values["boundary"],
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
        groups=groups[validation],
        doses=np.asarray([row["dose"] for row in parameter_rows]),
        clipping_fraction=float(np.mean(clipped)),
        maximum_new_boundary=float(np.max(arrays["boundary"])),
        fallback=fallback,
        gates=gates,
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
    )
    result.update(
        {
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "fit_groups": sorted(set(map(str, groups[fit]))),
            "validation_groups": sorted(set(map(str, groups[validation]))),
            "model_sha256": _parameter_model_identity(projection, model),
            "rows": output_rows,
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
    development = sorted(development_rows, key=lambda row: str(row["pair_id"]))
    confirmation = sorted(confirmation_rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in development]
    parameters = effective_case_parameters(oracle_report)
    all_rows = development + confirmation
    features = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in all_rows]
    )
    fit = np.arange(len(development), dtype=np.int64)
    query = np.arange(len(development), len(all_rows), dtype=np.int64)
    projection, projected, model, shuffled_model = _fit_models(
        features=features, parameters=parameters, fit=fit, model_spec=model_spec
    )
    development_groups = np.asarray(
        [str(row["group"]) for row in development], dtype=object
    )
    distances, _, _ = _standardized_distances(
        projected[fit], projected[query]
    )
    nearest_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(
        projected[fit],
        development_groups,
        float(evaluation["ood_distance_quantile"]),
    )
    fallback = nearest_distance > threshold
    raw = predict_parameters(model, projected[query])
    shuffled_raw = predict_parameters(shuffled_model, projected[query])
    transports, parameter_rows, clipped = _bounded_transports(
        raw, parameters, operator_config
    )
    shuffled_transports, _, _ = _bounded_transports(
        shuffled_raw, parameters, operator_config
    )
    pooled = TriangularLogitTransport(
        np.asarray(oracle_report["pooled_operator"]["parameters"], dtype=np.float64),
        dose=float(oracle_report["pooled_operator"]["dose"]),
    )
    oracle_rows = {str(row["pair_id"]): row for row in oracle_report["rows"]}
    nearest_rows = {str(row["pair_id"]): row for row in nearest_report["rows"]}
    samples = int(evaluation["confirmation_samples_per_image"])
    epsilon = float(evaluation["boundary_epsilon"])
    arrays: dict[str, list[float]] = {
        name: []
        for name in (
            "baseline", "selected", "identity", "strength", "nearest",
            "oracle", "shuffled", "target_style", "global_style",
            "selected_style", "boundary",
        )
    }
    output_rows = []
    for position, row in enumerate(confirmation):
        pair_id = str(row["pair_id"])
        evidence = oracle_rows[pair_id]
        source = _even_samples(_rgb(row["source"]), samples)
        target = _even_samples(_rgb(row["target"]), samples)
        global_output = pooled.apply(source)
        output = (
            global_output
            if fallback[position]
            else transports[position].apply(source)
        )
        shuffled_output = (
            global_output
            if fallback[position]
            else shuffled_transports[position].apply(source)
        )
        values = {
            "baseline": float(evidence["global_rmse"]),
            "selected": _rmse(output, target),
            "identity": float(evidence["identity_rmse"]),
            "strength": float(evidence["strength_oracle_rmse"]),
            "nearest": float(nearest_rows[pair_id]["selected_rmse"]),
            "oracle": float(evidence["case_oracle_rmse"]),
            "shuffled": _rmse(shuffled_output, target),
            "target_style": _rmse(target, source),
            "global_style": _rmse(global_output, source),
            "selected_style": _rmse(output, source),
            "boundary": _new_boundary_fraction(source, output, epsilon),
        }
        for name, value in values.items():
            arrays[name].append(value)
        output_rows.append(
            {
                "pair_id": pair_id,
                "group": str(row["group"]),
                **parameter_rows[position],
                "fallback": bool(fallback[position]),
                "distance": float(nearest_distance[position]),
                "threshold": threshold,
                "identity_rmse": values["identity"],
                "global_rmse": values["baseline"],
                "strength_oracle_rmse": values["strength"],
                "conditional_rmse": values["selected"],
                "nearest_rmse": values["nearest"],
                "oracle_rmse": values["oracle"],
                "new_boundary_fraction": values["boundary"],
            }
        )
    groups = np.asarray([str(row["group"]) for row in confirmation], dtype=object)
    result = _metrics(
        baseline=np.asarray(arrays["baseline"]), selected=np.asarray(arrays["selected"]),
        identity=np.asarray(arrays["identity"]), strength=np.asarray(arrays["strength"]),
        nearest=np.asarray(arrays["nearest"]), oracle=np.asarray(arrays["oracle"]),
        shuffled=np.asarray(arrays["shuffled"]), target_style=np.asarray(arrays["target_style"]),
        global_style=np.asarray(arrays["global_style"]), selected_style=np.asarray(arrays["selected_style"]),
        groups=groups, doses=np.asarray([row["dose"] for row in parameter_rows]),
        clipping_fraction=float(np.mean(clipped)), maximum_new_boundary=float(np.max(arrays["boundary"])),
        fallback=fallback,
        gates=gates, bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
    )
    result.update({"model_sha256": _parameter_model_identity(projection, model), "rows": output_rows})
    return result


__all__ = [
    "FiveKConditionalExplicitOperatorError",
    "effective_case_parameters",
    "evaluate_confirmation",
    "evaluate_development",
    "fit_parameter_ridge",
    "predict_parameters",
]
