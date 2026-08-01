"""Source-only factorized prediction of intrinsically bounded FiveK LUTs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_adaptive_lut_basis_development import _fit_predict_basis
from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import (
    _group_partition,
    combined_source_descriptor,
    fit_projection,
    project_features,
)
from src.eval.fivek_source_hard_retrieval import (
    _group_bootstrap,
    _source_only_threshold,
    _standardized_distances,
    source_descriptor,
)
from src.eval.fivek_strict_interior_lut_basis_development import (
    apply_strict_interior_lut,
)
from src.eval.fivek_strict_interior_lut_casebank import (
    _array_sha256,
    _new_boundary_fraction,
    fit_strict_interior_lut_bank,
)


class FiveKSourceAdaptiveLUTError(ValueError):
    """Raised when source-only LUT prediction violates the frozen protocol."""


def _apply(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return apply_strict_interior_lut(source[:, None, :], lut)[:, 0, :]


def _ratios(baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    return (
        float(np.quantile(candidate, 0.95) / max(np.quantile(baseline, 0.95), 1e-12)),
        float(np.max(candidate) / max(np.max(baseline), 1e-12)),
    )


def evaluate_source_adaptive_lut(
    *,
    rows: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    samples_per_image: int,
) -> tuple[dict[str, Any], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if not ordered:
        raise FiveKSourceAdaptiveLUTError("empty population")
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups,
        int(model_spec["group_bucket_modulus"]),
        int(model_spec["validation_bucket"]),
    )
    coefficients = fit_strict_interior_lut_bank(ordered, operator)

    combined = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in ordered]
    )
    mean, scale, components = fit_projection(
        combined[fit], int(descriptor_spec["pca_components"])
    )
    projected_fit = project_features(combined[fit], mean, scale, components)
    projected_validation = project_features(
        combined[validation], mean, scale, components
    )
    predicted, repeated_global, fit_evidence = _fit_predict_basis(
        projected_fit,
        coefficients[fit],
        groups[fit],
        projected_validation,
        rank=int(model_spec["lut_basis_rank"]),
        alphas=[float(value) for value in model_spec["ridge_alphas"]],
        maximum_absolute_residual=1.0,
    )
    adaptive_threshold = _source_only_threshold(
        projected_fit,
        groups[fit],
        float(model_spec["ood_distance_quantile"]),
    )
    adaptive_distance = np.min(
        _standardized_distances(projected_fit, projected_validation)[0], axis=1
    )
    adaptive_fallback = adaptive_distance > adaptive_threshold
    predicted[adaptive_fallback] = repeated_global[adaptive_fallback]

    tone = np.stack(
        [source_descriptor(row["source"], "tone_layout", descriptor_spec) for row in ordered]
    )
    nearest_threshold = _source_only_threshold(
        tone[fit], groups[fit], float(model_spec["ood_distance_quantile"])
    )
    nearest_distance_matrix = _standardized_distances(
        tone[fit], tone[validation]
    )[0]
    nearest_positions = np.argmin(nearest_distance_matrix, axis=1)
    nearest_distance = nearest_distance_matrix[
        np.arange(len(validation)), nearest_positions
    ]
    nearest_fallback = nearest_distance > nearest_threshold
    nearest_coefficients = coefficients[fit[nearest_positions]].copy()
    nearest_coefficients[nearest_fallback] = repeated_global[nearest_fallback]

    rng = np.random.default_rng(int(model_spec["shuffled_prediction_seed"]))
    shuffled = predicted[rng.permutation(len(predicted))].copy()
    shuffled[adaptive_fallback] = repeated_global[adaptive_fallback]

    strength_doses = [float(value) for value in evaluation["strength_doses"]]
    if (
        not strength_doses
        or strength_doses != sorted(set(strength_doses))
        or strength_doses[0] != 0.0
        or strength_doses[-1] != 1.0
    ):
        raise FiveKSourceAdaptiveLUTError("invalid strength controls")

    records = []
    arrays = {
        key: []
        for key in (
            "identity",
            "global",
            "adaptive",
            "nearest",
            "strength",
            "shuffled",
            "oracle",
        )
    }
    adaptive_styles = []
    global_styles = []
    boundary = []
    out_of_cube = []
    for position, index in enumerate(validation):
        source = _even_samples(np.asarray(ordered[index]["source"]), samples_per_image)
        target = _even_samples(np.asarray(ordered[index]["target"]), samples_per_image)
        identity_error = _rmse(source, target)
        global_output = _apply(source, repeated_global[position])
        adaptive_output = _apply(source, predicted[position])
        nearest_output = _apply(source, nearest_coefficients[position])
        shuffled_output = _apply(source, shuffled[position])
        strength_errors = [
            _rmse(_apply(source, repeated_global[position] * dose), target)
            for dose in strength_doses
        ]
        case_outputs = np.stack([_apply(source, lut) for lut in coefficients[fit]])
        case_errors = np.sqrt(
            np.mean((case_outputs - target[None, ...]) ** 2, axis=(1, 2))
        )
        oracle_position = int(np.argmin(case_errors))
        outputs = [global_output, adaptive_output, nearest_output, shuffled_output]
        errors = {
            "identity": identity_error,
            "global": _rmse(global_output, target),
            "adaptive": _rmse(adaptive_output, target),
            "nearest": _rmse(nearest_output, target),
            "strength": min(strength_errors),
            "shuffled": _rmse(shuffled_output, target),
            "oracle": float(case_errors[oracle_position]),
        }
        for name, value in errors.items():
            arrays[name].append(value)
        adaptive_style = _rmse(adaptive_output, source) / max(identity_error, 1e-12)
        global_style = _rmse(global_output, source) / max(identity_error, 1e-12)
        adaptive_styles.append(adaptive_style)
        global_styles.append(global_style)
        boundary.append(
            max(
                _new_boundary_fraction(source, output, float(operator["boundary_epsilon"]))
                for output in outputs
            )
        )
        out_of_cube.append(
            max(float(np.mean((output < 0.0) | (output > 1.0))) for output in outputs)
        )
        records.append(
            {
                "pair_id": ids[index],
                "group": str(groups[index]),
                **{f"{name}_rmse": value for name, value in errors.items()},
                "adaptive_target_style_retention": adaptive_style,
                "global_target_style_retention": global_style,
                "adaptive_fallback": bool(adaptive_fallback[position]),
                "adaptive_ood_distance": float(adaptive_distance[position]),
                "nearest_case_id": ids[fit[nearest_positions[position]]],
                "nearest_fallback": bool(nearest_fallback[position]),
                "nearest_ood_distance": float(nearest_distance[position]),
                "oracle_case_id": ids[fit[oracle_position]],
                "maximum_new_boundary_fraction": boundary[-1],
                "maximum_out_of_cube_fraction": out_of_cube[-1],
            }
        )
    values = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    global_mean = float(np.mean(values["global"]))
    adaptive_mean = float(np.mean(values["adaptive"]))
    oracle_gain = (global_mean - float(np.mean(values["oracle"]))) / max(global_mean, 1e-12)
    p95_ratio, worst_ratio = _ratios(values["global"], values["adaptive"])
    bootstrap = _group_bootstrap(
        values["global"],
        values["adaptive"],
        groups[validation],
        seed=int(evaluation["bootstrap_seed"]),
        repetitions=int(evaluation["bootstrap_repetitions"]),
    )

    def improvement(baseline: str) -> float:
        base = float(np.mean(values[baseline]))
        return (base - adaptive_mean) / max(base, 1e-12)

    metrics = {
        "input_rows": len(ordered),
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "coefficient_bank_sha256": _array_sha256(coefficients),
        "fit_groups": sorted(set(map(str, groups[fit]))),
        "validation_groups": sorted(set(map(str, groups[validation]))),
        "fit_evidence": fit_evidence,
        "adaptive_ood_threshold": adaptive_threshold,
        "nearest_ood_threshold": nearest_threshold,
        "fallback_fraction": float(np.mean(adaptive_fallback)),
        "nearest_fallback_fraction": float(np.mean(nearest_fallback)),
        "mean_improvement_over_global": improvement("global"),
        "win_fraction_over_global": float(np.mean(values["adaptive"] < values["global"])),
        "p95_ratio_to_global": p95_ratio,
        "worst_ratio_to_global": worst_ratio,
        "oracle_gap_closure": improvement("global") / max(oracle_gain, 1e-12),
        "mean_improvement_over_strength_oracle": improvement("strength"),
        "win_fraction_over_strength_oracle": float(
            np.mean(values["adaptive"] < values["strength"])
        ),
        "mean_improvement_over_nearest": improvement("nearest"),
        "win_fraction_over_nearest": float(np.mean(values["adaptive"] < values["nearest"])),
        "mean_improvement_over_shuffled": improvement("shuffled"),
        "bootstrap_improvement_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "median_target_style_retention": float(np.median(adaptive_styles)),
        "mean_style_magnitude_ratio_to_global": float(
            np.mean(adaptive_styles) / max(np.mean(global_styles), 1e-12)
        ),
        "maximum_new_boundary_fraction": float(max(boundary)),
        "maximum_out_of_cube_fraction": float(max(out_of_cube)),
    }
    gates = {
        "mean": metrics["mean_improvement_over_global"]
        >= float(evaluation["minimum_mean_improvement_over_global"]),
        "wins": metrics["win_fraction_over_global"]
        >= float(evaluation["minimum_win_fraction_over_global"]),
        "p95": metrics["p95_ratio_to_global"]
        <= float(evaluation["maximum_p95_ratio_to_global"]),
        "worst": metrics["worst_ratio_to_global"]
        <= float(evaluation["maximum_worst_ratio_to_global"]),
        "oracle_gap": metrics["oracle_gap_closure"]
        >= float(evaluation["minimum_oracle_gap_closure"]),
        "strength_mean": metrics["mean_improvement_over_strength_oracle"]
        >= float(evaluation["minimum_mean_improvement_over_strength_oracle"]),
        "strength_wins": metrics["win_fraction_over_strength_oracle"]
        >= float(evaluation["minimum_win_fraction_over_strength_oracle"]),
        "nearest_mean": metrics["mean_improvement_over_nearest"]
        >= float(evaluation["minimum_mean_improvement_over_nearest"]),
        "nearest_wins": metrics["win_fraction_over_nearest"]
        >= float(evaluation["minimum_win_fraction_over_nearest"]),
        "shuffle": metrics["mean_improvement_over_shuffled"]
        >= float(evaluation["minimum_mean_improvement_over_shuffled"]),
        "bootstrap": metrics["bootstrap_improvement_ci95"][0]
        > float(evaluation["minimum_bootstrap_lower_improvement"]),
        "style": metrics["median_target_style_retention"]
        >= float(evaluation["minimum_median_target_style_retention"]),
        "style_vs_global": metrics["mean_style_magnitude_ratio_to_global"]
        >= float(evaluation["minimum_mean_style_magnitude_ratio_to_global"]),
        "fallback": metrics["fallback_fraction"]
        <= float(evaluation["maximum_fallback_fraction"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(evaluation["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"]
        <= float(evaluation["maximum_out_of_cube_fraction"]),
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "rows": records,
    }, coefficients


__all__ = [
    "FiveKSourceAdaptiveLUTError",
    "evaluate_source_adaptive_lut",
]
