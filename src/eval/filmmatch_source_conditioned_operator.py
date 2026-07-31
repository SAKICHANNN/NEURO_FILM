"""Source-conditioned bounded explicit operators for the FilmMatch session.

The predictor sees only source-chart statistics.  Paired targets are used to
fit development group operators and evaluator Oracles, never as predictor
features.  Every outer fold holds out one complete illuminant.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurvePositiveMatrixOperator,
    fit_monotone_curve_positive_matrix,
)
from src.roll2film.positive_film_fitting import (
    row_stochastic_identity_mixture,
)


def _softmax_with_zero_reference(logits: np.ndarray) -> np.ndarray:
    values = np.zeros((3, 3), dtype=np.float64)
    values[:, 1:] = np.asarray(logits, dtype=np.float64).reshape(3, 2)
    values -= np.max(values, axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def encode_operator_parameters(
    operator: MonotoneCurvePositiveMatrixOperator,
    *,
    curve_identity_mixture: float,
    matrix_identity_mixture: float,
) -> np.ndarray:
    """Recover the twelve gauge-fixed logits from a decoded operator."""

    curve_probability = (
        operator.curve_segment_weights
        - (1.0 - curve_identity_mixture) / 3.0
    ) / curve_identity_mixture
    matrix_probability = (
        operator.matrix
        - (1.0 - matrix_identity_mixture) * np.eye(3)
    ) / matrix_identity_mixture
    if (
        np.any(curve_probability <= 0.0)
        or np.any(matrix_probability <= 0.0)
        or not np.all(np.isfinite(curve_probability))
        or not np.all(np.isfinite(matrix_probability))
    ):
        raise ValueError("operator lies outside the configured parameterization")
    curve_logits = np.log(
        curve_probability[:, 1:] / curve_probability[:, :1]
    ).reshape(-1)
    matrix_logits = []
    for row in range(3):
        for column in range(3):
            if column != row:
                matrix_logits.append(
                    np.log(
                        matrix_probability[row, column]
                        / matrix_probability[row, row]
                    )
                )
    return np.concatenate((curve_logits, np.asarray(matrix_logits)))


def decode_operator_parameters(
    parameters: np.ndarray,
    *,
    curve_identity_mixture: float,
    matrix_identity_mixture: float,
    free_logit_bounds: tuple[float, float],
) -> tuple[MonotoneCurvePositiveMatrixOperator, int]:
    """Clip to the preregistered parameter box and decode a safe operator."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain twelve finite values")
    lower, upper = map(float, free_logit_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("invalid free-logit bounds")
    clipped = np.clip(values, lower, upper)
    clip_count = int(np.count_nonzero(clipped != values))
    curve_probability = _softmax_with_zero_reference(clipped[:6])
    weights = (
        (1.0 - curve_identity_mixture) / 3.0
        + curve_identity_mixture * curve_probability
    )
    matrix = row_stochastic_identity_mixture(
        clipped[6:], identity_mixture=matrix_identity_mixture
    )
    return MonotoneCurvePositiveMatrixOperator(weights, matrix), clip_count


def source_descriptor(
    source_rgb: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    """Compute a content-fixed, source-only exposure/colour-state descriptor."""

    source = np.asarray(source_rgb, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or source.shape[0] < 12
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("source_rgb must be finite [0,1] Nx3")
    descriptor = config["source_descriptor"]
    rgb_q = np.asarray(descriptor["rgb_quantiles"], dtype=np.float64)
    luma_q = np.asarray(descriptor["luma_quantiles"], dtype=np.float64)
    chroma_q = np.asarray(descriptor["chroma_quantiles"], dtype=np.float64)
    chromaticity_q = np.asarray(
        descriptor["chromaticity_quantiles"], dtype=np.float64
    )
    for quantiles in (rgb_q, luma_q, chroma_q, chromaticity_q):
        if (
            quantiles.ndim != 1
            or not len(quantiles)
            or np.any(quantiles <= 0.0)
            or np.any(quantiles >= 1.0)
        ):
            raise ValueError("descriptor quantiles must lie strictly in (0,1)")
    luma = source @ np.asarray([0.2126, 0.7152, 0.0722])
    chroma = np.max(source, axis=1) - np.min(source, axis=1)
    total = np.sum(source, axis=1) + float(descriptor["epsilon"])
    chromaticity = source[:, :2] / total[:, None]
    values = np.concatenate(
        (
            np.quantile(source, rgb_q, axis=0).reshape(-1),
            np.quantile(luma, luma_q),
            np.quantile(chromaticity, chromaticity_q, axis=0).reshape(-1),
            np.quantile(chroma, chroma_q),
        )
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("source descriptor is non-finite")
    return values


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, *, alpha: float
) -> dict[str, np.ndarray | float]:
    """Fit deterministic standardized multi-output ridge with an intercept."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError("ridge needs matching finite matrices and alpha > 0")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x - mean) / scale
    target_mean = np.mean(y, axis=0)
    centered = y - target_mean
    gram = normalized.T @ normalized
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        normalized.T @ centered,
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": target_mean,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def predict_ridge(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    prediction = (
        (x - np.asarray(model["feature_mean"]))
        / np.asarray(model["feature_scale"])
    ) @ np.asarray(model["coefficients"]) + np.asarray(model["target_mean"])
    if not np.all(np.isfinite(prediction)):
        raise ValueError("ridge prediction is non-finite")
    return prediction


def select_nested_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    illuminants: np.ndarray,
    candidates: list[float],
) -> tuple[float, list[dict[str, float]]]:
    """Select alpha with leave-one-development-illuminant-out prediction."""

    labels = np.asarray(illuminants)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("nested alpha selection needs at least two illuminants")
    rows = []
    for alpha in map(float, candidates):
        errors = []
        for held in unique:
            train = labels != held
            validation = ~train
            model = fit_ridge(features[train], targets[train], alpha=alpha)
            residual = predict_ridge(model, features[validation]) - targets[validation]
            errors.append(float(np.sqrt(np.mean(np.square(residual)))))
        rows.append(
            {
                "alpha": alpha,
                "mean_parameter_rmse": float(np.mean(errors)),
                "worst_parameter_rmse": float(np.max(errors)),
            }
        )
    selected = min(rows, key=lambda row: (row["mean_parameter_rmse"], -row["alpha"]))
    return float(selected["alpha"]), rows


def _fit_group_operator(
    source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
) -> MonotoneCurvePositiveMatrixOperator:
    operator = config["operator"]
    fit = operator["group_fit"]
    return fit_monotone_curve_positive_matrix(
        source,
        target,
        curve_identity_mixture=float(operator["curve_identity_mixture"]),
        matrix_identity_mixture=float(operator["matrix_identity_mixture"]),
        free_logit_bounds=tuple(map(float, operator["free_logit_bounds"])),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(fit["maximum_function_evaluations"]),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    ).operator


def _cube(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _aggregate_predictions(
    predictions: list[np.ndarray], targets: list[np.ndarray]
) -> dict[str, float]:
    return prediction_metrics(np.concatenate(predictions), np.concatenate(targets))


def evaluate_source_conditioned_operator(
    datasets: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = datasets["reflective_records"]
    group_ids = np.asarray(
        [
            f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}"
            for row in records
        ]
    )
    row_illuminants = np.asarray([str(row["illuminant"]) for row in records])
    groups = sorted(set(group_ids.tolist()))
    if len(groups) != int(config["data"]["groups"]):
        raise ValueError("reflective group count drift")
    illuminant_by_group = {
        group: str(row_illuminants[np.flatnonzero(group_ids == group)[0]])
        for group in groups
    }
    expected_patches = int(config["data"]["patches_per_group"])
    group_source = {}
    group_target = {}
    descriptors = {}
    fitted_operators = {}
    fitted_parameters = {}
    operator_config = config["operator"]
    for group in groups:
        selected = group_ids == group
        if int(np.count_nonzero(selected)) != expected_patches:
            raise ValueError("reflective patches-per-group drift")
        group_source[group] = source[selected]
        group_target[group] = target[selected]
        descriptors[group] = source_descriptor(source[selected], config)
        fitted_operators[group] = _fit_group_operator(
            source[selected], target[selected], config
        )
        fitted_parameters[group] = encode_operator_parameters(
            fitted_operators[group],
            curve_identity_mixture=float(operator_config["curve_identity_mixture"]),
            matrix_identity_mixture=float(operator_config["matrix_identity_mixture"]),
        )

    outer_illuminants = sorted(set(illuminant_by_group.values()))
    fold_rows = []
    all_predictions: dict[str, list[np.ndarray]] = {
        "global": [],
        "adaptive": [],
        "oracle": [],
        "shuffled": [],
    }
    all_targets: list[np.ndarray] = []
    total_clipped = 0
    total_predicted_logits = 0
    cube = _cube(int(config["structural_audit"]["cube_size"]))
    minimum_jacobian = np.inf
    maximum_oog = 0.0
    alpha_candidates = [float(value) for value in config["predictor"]["alpha_candidates"]]
    seed = int(config["predictor"]["seed"])

    for outer_index, held_illuminant in enumerate(outer_illuminants):
        development_groups = [
            group
            for group in groups
            if illuminant_by_group[group] != held_illuminant
        ]
        held_groups = [
            group
            for group in groups
            if illuminant_by_group[group] == held_illuminant
        ]
        x_train = np.stack([descriptors[group] for group in development_groups])
        y_train = np.stack([fitted_parameters[group] for group in development_groups])
        train_illuminants = np.asarray(
            [illuminant_by_group[group] for group in development_groups]
        )
        alpha, alpha_rows = select_nested_alpha(
            x_train, y_train, train_illuminants, alpha_candidates
        )
        model = fit_ridge(x_train, y_train, alpha=alpha)
        rng = np.random.default_rng(seed + outer_index)
        shuffled_targets = y_train[rng.permutation(len(y_train))]
        shuffled_alpha, shuffled_alpha_rows = select_nested_alpha(
            x_train, shuffled_targets, train_illuminants, alpha_candidates
        )
        shuffled_model = fit_ridge(
            x_train, shuffled_targets, alpha=shuffled_alpha
        )
        development_source = np.concatenate(
            [group_source[group] for group in development_groups]
        )
        development_target = np.concatenate(
            [group_target[group] for group in development_groups]
        )
        global_operator = _fit_group_operator(
            development_source, development_target, config
        )
        group_rows = []
        for group in held_groups:
            predicted_parameters = predict_ridge(model, descriptors[group])[0]
            adaptive_operator, clipped = decode_operator_parameters(
                predicted_parameters,
                curve_identity_mixture=float(operator_config["curve_identity_mixture"]),
                matrix_identity_mixture=float(operator_config["matrix_identity_mixture"]),
                free_logit_bounds=tuple(map(float, operator_config["free_logit_bounds"])),
            )
            shuffled_parameters = predict_ridge(
                shuffled_model, descriptors[group]
            )[0]
            shuffled_operator, shuffled_clipped = decode_operator_parameters(
                shuffled_parameters,
                curve_identity_mixture=float(operator_config["curve_identity_mixture"]),
                matrix_identity_mixture=float(operator_config["matrix_identity_mixture"]),
                free_logit_bounds=tuple(map(float, operator_config["free_logit_bounds"])),
            )
            total_clipped += clipped
            total_predicted_logits += 12
            predictions = {
                "global": global_operator.apply(group_source[group]),
                "adaptive": adaptive_operator.apply(group_source[group]),
                "oracle": fitted_operators[group].apply(group_source[group]),
                "shuffled": shuffled_operator.apply(group_source[group]),
            }
            metrics = {
                name: prediction_metrics(value, group_target[group])
                for name, value in predictions.items()
            }
            determinants = adaptive_operator.jacobian_determinants(cube)
            minimum_jacobian = min(minimum_jacobian, float(np.min(determinants)))
            maximum_oog = max(
                maximum_oog,
                float(
                    np.mean(
                        np.any(
                            (predictions["adaptive"] < 0.0)
                            | (predictions["adaptive"] > 1.0),
                            axis=1,
                        )
                    )
                ),
            )
            group_rows.append(
                {
                    "group": group,
                    "metrics": metrics,
                    "adaptive_improvement_over_global": float(
                        1.0
                        - metrics["adaptive"]["rgb_rmse"]
                        / metrics["global"]["rgb_rmse"]
                    ),
                    "oracle_improvement_over_global": float(
                        1.0
                        - metrics["oracle"]["rgb_rmse"]
                        / metrics["global"]["rgb_rmse"]
                    ),
                    "adaptive_improvement_over_shuffled": float(
                        1.0
                        - metrics["adaptive"]["rgb_rmse"]
                        / metrics["shuffled"]["rgb_rmse"]
                    ),
                    "predicted_logit_clip_count": clipped,
                    "shuffled_logit_clip_count": shuffled_clipped,
                    "minimum_cube_jacobian": float(np.min(determinants)),
                }
            )
            for name, value in predictions.items():
                all_predictions[name].append(value)
            all_targets.append(group_target[group])
        fold_rows.append(
            {
                "held_illuminant": held_illuminant,
                "development_groups": len(development_groups),
                "held_groups": len(held_groups),
                "selected_alpha": alpha,
                "nested_alpha_scores": alpha_rows,
                "shuffled_selected_alpha": shuffled_alpha,
                "shuffled_nested_alpha_scores": shuffled_alpha_rows,
                "groups": group_rows,
            }
        )

    aggregate_metrics = {
        name: _aggregate_predictions(values, all_targets)
        for name, values in all_predictions.items()
    }
    global_rmse = aggregate_metrics["global"]["rgb_rmse"]
    adaptive_rmse = aggregate_metrics["adaptive"]["rgb_rmse"]
    oracle_rmse = aggregate_metrics["oracle"]["rgb_rmse"]
    shuffled_rmse = aggregate_metrics["shuffled"]["rgb_rmse"]
    group_improvements = np.asarray(
        [
            row["adaptive_improvement_over_global"]
            for fold in fold_rows
            for row in fold["groups"]
        ]
    )
    aggregate = {
        "metrics": aggregate_metrics,
        "oracle_rgb_rmse_improvement_over_global": float(1.0 - oracle_rmse / global_rmse),
        "adaptive_rgb_rmse_improvement_over_global": float(1.0 - adaptive_rmse / global_rmse),
        "adaptive_group_wins": int(np.count_nonzero(group_improvements > 0.0)),
        "adaptive_group_win_fraction": float(np.mean(group_improvements > 0.0)),
        "adaptive_p95_ratio_to_global": float(
            aggregate_metrics["adaptive"]["p95_rgb_euclidean"]
            / aggregate_metrics["global"]["p95_rgb_euclidean"]
        ),
        "adaptive_rgb_rmse_improvement_over_shuffled": float(
            1.0 - adaptive_rmse / shuffled_rmse
        ),
        "predicted_logit_clip_fraction": float(total_clipped / total_predicted_logits),
        "minimum_cube_jacobian": float(minimum_jacobian),
        "maximum_out_of_cube_fraction": float(maximum_oog),
    }
    gate = config["automatic_gate"]
    structural = config["structural_audit"]
    gate_results = {
        "oracle_value": aggregate["oracle_rgb_rmse_improvement_over_global"]
        >= float(gate["minimum_oracle_mean_rgb_rmse_improvement_over_global"]),
        "adaptive_value": aggregate["adaptive_rgb_rmse_improvement_over_global"]
        >= float(gate["minimum_adaptive_mean_rgb_rmse_improvement_over_global"]),
        "group_wins": aggregate["adaptive_group_win_fraction"]
        >= float(gate["minimum_adaptive_group_win_fraction"]),
        "p95": aggregate["adaptive_p95_ratio_to_global"]
        <= float(gate["maximum_adaptive_aggregate_p95_ratio_to_global"]),
        "shuffled_control": aggregate["adaptive_rgb_rmse_improvement_over_shuffled"]
        >= float(gate["minimum_adaptive_mean_rgb_rmse_improvement_over_shuffled"]),
        "logit_support": aggregate["predicted_logit_clip_fraction"]
        <= float(gate["maximum_predicted_logit_clip_fraction"]),
        "out_of_cube": aggregate["maximum_out_of_cube_fraction"]
        <= float(structural["maximum_out_of_cube_fraction"]),
        "jacobian": aggregate["minimum_cube_jacobian"]
        >= float(structural["minimum_jacobian_determinant"]),
    }
    passed = bool(all(gate_results.values()))
    if not gate_results["oracle_value"]:
        branch = "no_oracle"
    elif not gate_results["shuffled_control"]:
        branch = "shuffled_or_unstable"
    elif not gate_results["out_of_cube"] or not gate_results["jacobian"]:
        branch = "structural_failure"
    elif passed:
        branch = "pass"
    else:
        branch = "oracle_only"
    report = {
        "schema": config["schema"],
        "experiment_id": config["experiment_id"],
        "folds": fold_rows,
        "aggregate": aggregate,
        "gate_results": gate_results,
        "automatic_gate_passed": passed,
        "branch": branch,
        "validation_render_opened": False,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "decode_operator_parameters",
    "encode_operator_parameters",
    "evaluate_source_conditioned_operator",
    "fit_ridge",
    "predict_ridge",
    "select_nested_alpha",
    "source_descriptor",
]
