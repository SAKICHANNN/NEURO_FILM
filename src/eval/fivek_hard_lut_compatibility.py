"""Source-only hard Top-1 compatibility routing over intact bounded LUTs."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import (
    _deterministic_random_cases,
    _group_partition,
    _metrics,
    _model_identity,
    combined_source_descriptor,
    fit_projection,
    fit_ridge_ranker,
    project_features,
    rank_queries,
    training_matrix,
)
from src.eval.fivek_source_hard_retrieval import (
    _source_only_threshold,
    _standardized_distances,
    source_descriptor,
)
from src.eval.fivek_strict_interior_lut_basis_development import (
    apply_strict_interior_lut,
)
from src.eval.fivek_strict_interior_lut_casebank import (
    _apply_bank,
    _new_boundary_fraction,
    fit_strict_interior_lut_bank,
)


class FiveKHardLUTCompatibilityError(ValueError):
    """Raised when the hard LUT router violates its frozen protocol."""


def _apply(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return apply_strict_interior_lut(source[:, None, :], lut)[:, 0, :]


def _compatibility_matrix(
    errors: np.ndarray, styles: np.ndarray, minimum_style: float
) -> np.ndarray:
    """Rank all style-eligible cases ahead of bland cases for each query."""

    error = np.asarray(errors, dtype=np.float64)
    style = np.asarray(styles, dtype=np.float64)
    if error.shape != style.shape or error.ndim != 2:
        raise FiveKHardLUTCompatibilityError("invalid compatibility arrays")
    span = np.ptp(error, axis=1, keepdims=True) + 1.0e-12
    penalty = style < float(minimum_style)
    return error + penalty * (2.0 * span + 1.0e-12)


def evaluate_hard_lut_compatibility(
    *,
    rows: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    selector_spec: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    samples_per_image: int,
) -> tuple[dict[str, Any], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups,
        int(model_spec["group_bucket_modulus"]),
        int(model_spec["validation_bucket"]),
    )
    coefficients = fit_strict_interior_lut_bank(ordered, operator)
    error_matrix = np.empty((len(ordered), len(ordered)), dtype=np.float64)
    style_matrix = np.empty_like(error_matrix)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for query, row in enumerate(ordered):
        source = _even_samples(np.asarray(row["source"]), samples_per_image)
        target = _even_samples(np.asarray(row["target"]), samples_per_image)
        outputs = _apply_bank(source, coefficients)
        error_matrix[query] = np.sqrt(
            np.mean((outputs - target[None, ...]) ** 2, axis=(1, 2))
        )
        identity_error = _rmse(source, target)
        style_matrix[query] = np.sqrt(
            np.mean((outputs - source[None, ...]) ** 2, axis=(1, 2))
        ) / max(identity_error, 1.0e-12)
        sources.append(source)
        targets.append(target)
    compatibility = _compatibility_matrix(
        error_matrix,
        style_matrix,
        float(evaluation["minimum_case_target_style_retention"]),
    )

    features = np.stack(
        [combined_source_descriptor(row["source"], descriptor_spec) for row in ordered]
    )
    projection = fit_projection(features[fit], int(model_spec["pca_components"]))
    projected = project_features(features, *projection)
    x, y = training_matrix(projected, fit, compatibility)
    model = fit_ridge_ranker(x, y, float(model_spec["ridge_alpha"]))
    sx, sy = training_matrix(
        projected,
        fit,
        compatibility,
        shuffled_seed=int(model_spec["shuffled_label_seed"]),
    )
    shuffled_model = fit_ridge_ranker(sx, sy, float(model_spec["ridge_alpha"]))
    selected_case = rank_queries(
        projected=projected,
        query_indices=validation,
        case_indices=fit,
        model=model,
    )
    shuffled_case = rank_queries(
        projected=projected,
        query_indices=validation,
        case_indices=fit,
        model=shuffled_model,
    )
    distances = _standardized_distances(projected[fit], projected[validation])[0]
    ood_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(
        projected[fit], groups[fit], float(selector_spec["ood_distance_quantile"])
    )
    fallback = ood_distance > threshold

    tone = np.stack(
        [source_descriptor(row["source"], "tone_layout", descriptor_spec) for row in ordered]
    )
    tone_distances = _standardized_distances(tone[fit], tone[validation])[0]
    nearest_positions = np.argmin(tone_distances, axis=1)
    nearest_threshold = _source_only_threshold(
        tone[fit], groups[fit], float(selector_spec["ood_distance_quantile"])
    )
    nearest_fallback = np.min(tone_distances, axis=1) > nearest_threshold
    global_lut = np.median(coefficients[fit], axis=0)
    random_case = _deterministic_random_cases(
        ids,
        [ids[index] for index in validation],
        fit,
        int(selector_spec["random_case_seed"]),
    )
    strength_doses = [float(value) for value in evaluation["strength_doses"]]

    arrays = {name: [] for name in ("global", "selected", "nearest", "oracle", "random", "shuffled", "strength")}
    selected_styles = []
    selected_case_eligible = []
    boundaries = []
    out_of_cube = []
    records = []
    for position, index in enumerate(validation):
        source = sources[index]
        target = targets[index]
        global_output = _apply(source, global_lut)
        global_error = _rmse(global_output, target)
        selected_index = int(selected_case[position])
        shuffled_index = int(shuffled_case[position])
        nearest_index = int(fit[nearest_positions[position]])
        eligible_fit = fit[
            style_matrix[index, fit]
            >= float(evaluation["minimum_case_target_style_retention"])
        ]
        if len(eligible_fit) == 0:
            raise FiveKHardLUTCompatibilityError("held row has no style-eligible case")
        oracle_index = int(eligible_fit[np.argmin(error_matrix[index, eligible_fit])])
        selected_error = global_error if fallback[position] else float(error_matrix[index, selected_index])
        shuffled_error = global_error if fallback[position] else float(error_matrix[index, shuffled_index])
        nearest_error = global_error if nearest_fallback[position] else float(error_matrix[index, nearest_index])
        selected_style = (
            _rmse(global_output, source) / max(_rmse(source, target), 1.0e-12)
            if fallback[position]
            else float(style_matrix[index, selected_index])
        )
        strength_error = min(
            _rmse(_apply(source, global_lut * dose), target) for dose in strength_doses
        )
        outputs = [
            global_output,
            _apply(source, coefficients[selected_index]),
            _apply(source, coefficients[nearest_index]),
        ]
        values = {
            "global": global_error,
            "selected": selected_error,
            "nearest": nearest_error,
            "oracle": float(error_matrix[index, oracle_index]),
            "random": float(error_matrix[index, int(random_case[position])]),
            "shuffled": shuffled_error,
            "strength": strength_error,
        }
        for name, value in values.items():
            arrays[name].append(value)
        selected_styles.append(selected_style)
        selected_case_eligible.append(
            bool(style_matrix[index, selected_index] >= float(evaluation["minimum_case_target_style_retention"]))
        )
        boundaries.append(
            max(
                _new_boundary_fraction(source, output, float(operator["boundary_epsilon"]))
                for output in outputs
            )
        )
        out_of_cube.append(
            max(float(np.mean((output < 0.0) | (output > 1.0))) for output in outputs)
        )
        records.append({
            "pair_id": ids[index],
            "group": str(groups[index]),
            "selected_case_id": ids[selected_index],
            "selected_case_style_eligible": selected_case_eligible[-1],
            "selected_target_style_retention": selected_style,
            "fallback": bool(fallback[position]),
            "distance": float(ood_distance[position]),
            "threshold": threshold,
            "nearest_case_id": ids[nearest_index],
            "oracle_case_id": ids[oracle_index],
            **{f"{name}_rmse": value for name, value in values.items()},
        })
    values = {name: np.asarray(array, dtype=np.float64) for name, array in arrays.items()}
    router_gate_keys = {
        "minimum_mean_improvement_over_global",
        "minimum_win_fraction_over_global",
        "maximum_p95_ratio_to_global",
        "maximum_worst_ratio_to_global",
        "minimum_oracle_gap_closure",
        "minimum_mean_improvement_over_nearest",
        "minimum_win_fraction_over_nearest",
        "minimum_mean_improvement_over_random_case",
        "minimum_mean_improvement_over_shuffled_ranker",
        "minimum_bootstrap_lower_improvement",
        "maximum_fallback_fraction",
        "minimum_distinct_selected_cases",
        "maximum_selected_case_share",
    }
    base = _metrics(
        baseline=values["global"],
        selected=values["selected"],
        nearest=values["nearest"],
        oracle=values["oracle"],
        random=values["random"],
        shuffled=values["shuffled"],
        groups=groups[validation],
        selected_ids=[ids[index] for index in selected_case],
        fallback=fallback,
        gates={key: evaluation[key] for key in router_gate_keys},
        selector=selector_spec,
    )
    selected_mean = float(np.mean(values["selected"]))
    strength_mean = float(np.mean(values["strength"]))
    additions = {
        "mean_improvement_over_strength_oracle": (strength_mean - selected_mean) / max(strength_mean, 1e-12),
        "win_fraction_over_strength_oracle": float(np.mean(values["selected"] < values["strength"])),
        "median_target_style_retention": float(np.median(selected_styles)),
        "selected_case_style_eligible_fraction": float(np.mean(selected_case_eligible)),
        "maximum_new_boundary_fraction": float(max(boundaries)),
        "maximum_out_of_cube_fraction": float(max(out_of_cube)),
    }
    base["metrics"].update(additions)
    base["gates"].update({
        "strength_mean": additions["mean_improvement_over_strength_oracle"] >= float(evaluation["minimum_mean_improvement_over_strength_oracle"]),
        "strength_wins": additions["win_fraction_over_strength_oracle"] >= float(evaluation["minimum_win_fraction_over_strength_oracle"]),
        "style": additions["median_target_style_retention"] >= float(evaluation["minimum_median_target_style_retention"]),
        "style_eligible": additions["selected_case_style_eligible_fraction"] >= float(evaluation["minimum_selected_case_style_eligible_fraction"]),
        "boundary": additions["maximum_new_boundary_fraction"] <= float(evaluation["maximum_new_boundary_fraction"]),
        "cube": additions["maximum_out_of_cube_fraction"] <= float(evaluation["maximum_out_of_cube_fraction"]),
    })
    base.update({
        "automatic_pass": all(base["gates"].values()),
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "fit_groups": sorted(set(map(str, groups[fit]))),
        "validation_groups": sorted(set(map(str, groups[validation]))),
        "model_sha256": _model_identity(projection, model),
        "rows": records,
    })
    return base, coefficients


__all__ = [
    "FiveKHardLUTCompatibilityError",
    "evaluate_hard_lut_compatibility",
]
