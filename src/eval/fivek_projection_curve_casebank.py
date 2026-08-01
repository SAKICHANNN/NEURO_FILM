"""Mature projection-curve capacity baseline on the large FiveK case bank."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import _group_partition
from src.eval.fivek_source_hard_retrieval import _rgb
from src.roll2film.parallel_projection_curves import (
    ParallelProjectionCurveOperator,
    fit_projection_curve_coefficients,
    select_safe_dose,
    validate_projection_directions,
)


class FiveKProjectionCurveCasebankError(ValueError):
    """Raised when the mature projection-curve baseline drifts."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, epsilon: float
) -> float:
    source_boundary = np.any(
        (source <= epsilon) | (source >= 1.0 - epsilon), axis=1
    )
    output_boundary = np.any(
        (output <= epsilon) | (output >= 1.0 - epsilon), axis=1
    )
    return float(np.mean(output_boundary & ~source_boundary))


def _safe_operator(
    coefficients: np.ndarray,
    operator: Mapping[str, Any],
) -> tuple[ParallelProjectionCurveOperator, dict[str, Any]]:
    return select_safe_dose(
        directions=np.asarray(operator["projection_directions"], dtype=np.float64),
        coefficients=coefficients,
        boundary_epsilon=float(operator["boundary_epsilon"]),
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(
            operator["minimum_jacobian_determinant"]
        ),
        maximum_jacobian_condition=float(
            operator["maximum_jacobian_condition"]
        ),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
        strict_interior_safety_factor=float(
            operator["strict_interior_safety_factor"]
        ),
    )


def fit_projection_curve_bank(
    rows: Sequence[Mapping[str, Any]], operator: Mapping[str, Any]
) -> np.ndarray:
    directions = validate_projection_directions(
        np.asarray(operator["projection_directions"], dtype=np.float64)
    )
    coefficients = [
        fit_projection_curve_coefficients(
            _rgb(row["source"]),
            _rgb(row["target"]),
            directions=directions,
            control_point_count=int(operator["control_point_count"]),
            boundary_epsilon=float(operator["boundary_epsilon"]),
            sample_stride=int(operator["sample_stride"]),
            target_activation_limit=float(operator["target_activation_limit"]),
            identity_shrinkage=float(operator["identity_shrinkage"]),
            first_difference_smoothness=float(
                operator["first_difference_smoothness"]
            ),
            coefficient_absolute_limit=float(
                operator["coefficient_absolute_limit"]
            ),
            strict_interior_safety_factor=float(
                operator["strict_interior_safety_factor"]
            ),
        )
        for row in rows
    ]
    result = np.stack(coefficients)
    if not np.all(np.isfinite(result)):
        raise FiveKProjectionCurveCasebankError("non-finite coefficient bank")
    return result


def projection_curve_fit_eligible(
    source: np.ndarray, operator: Mapping[str, Any]
) -> tuple[bool, int]:
    """Mirror the mature primitive's source-only minimum-system requirement."""

    rgb = _rgb(source)
    sampled = rgb[
        :: int(operator["sample_stride"]),
        :: int(operator["sample_stride"]),
    ].reshape(-1, 3)
    epsilon = float(operator["boundary_epsilon"])
    headroom = np.maximum(
        0.0, np.minimum(sampled - epsilon, 1.0 - epsilon - sampled)
    )
    valid = int(np.sum(np.all(headroom > 1.0e-12, axis=1)))
    required = int(operator["control_point_count"]) * len(
        operator["projection_directions"]
    )
    return valid >= required, valid


def evaluate_projection_curve_casebank(
    *,
    rows: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    split_spec: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    samples_per_image: int,
) -> tuple[dict[str, Any], np.ndarray]:
    all_rows = sorted(rows, key=lambda row: str(row["pair_id"]))
    eligibility = [
        projection_curve_fit_eligible(row["source"], operator)
        for row in all_rows
    ]
    ordered = [
        row for row, (eligible, _) in zip(all_rows, eligibility, strict=True)
        if eligible
    ]
    eligible_fraction = len(ordered) / len(all_rows)
    all_groups = {str(row["group"]) for row in all_rows}
    eligible_groups = {str(row["group"]) for row in ordered}
    if (
        eligible_fraction < float(evaluation["minimum_fit_eligible_fraction"])
        or (
            evaluation["require_all_source_groups_after_fit_eligibility"]
            and eligible_groups != all_groups
        )
    ):
        raise FiveKProjectionCurveCasebankError(
            "projection-curve fit eligibility gate failed"
        )
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups,
        int(split_spec["group_bucket_modulus"]),
        int(split_spec["validation_bucket"]),
    )
    coefficients = fit_projection_curve_bank(ordered, operator)
    operators_and_diagnostics = [
        _safe_operator(value, operator) for value in coefficients
    ]
    operators = [item[0] for item in operators_and_diagnostics]
    diagnostics = [item[1] for item in operators_and_diagnostics]
    global_coefficients = np.median(coefficients[fit], axis=0)
    global_operator, global_diagnostics = _safe_operator(
        global_coefficients, operator
    )
    epsilon = float(operator["boundary_epsilon"])
    style_threshold = float(
        evaluation["minimum_self_fit_median_style_retention"]
    )
    output_rows: list[dict[str, Any]] = []
    for index in validation:
        source = _even_samples(_rgb(ordered[index]["source"]), samples_per_image)
        target = _even_samples(_rgb(ordered[index]["target"]), samples_per_image)
        identity_error = _rmse(source, target)
        target_style = identity_error
        global_output = global_operator.apply(source)
        global_error = _rmse(global_output, target)
        self_output = operators[index].apply(source)
        self_error = _rmse(self_output, target)
        self_style = _rmse(self_output, source) / max(target_style, 1.0e-12)
        case_outputs = [operators[case].apply(source) for case in fit]
        case_errors = np.asarray(
            [_rmse(output, target) for output in case_outputs], dtype=np.float64
        )
        case_style = np.asarray(
            [
                _rmse(output, source) / max(target_style, 1.0e-12)
                for output in case_outputs
            ],
            dtype=np.float64,
        )
        oracle_position = int(np.argmin(case_errors))
        eligible = np.flatnonzero(case_style >= style_threshold)
        constrained_position = (
            int(eligible[np.argmin(case_errors[eligible])])
            if len(eligible)
            else None
        )
        selected_outputs = [global_output, self_output, case_outputs[oracle_position]]
        if constrained_position is not None:
            selected_outputs.append(case_outputs[constrained_position])
        output_rows.append(
            {
                "pair_id": ids[index],
                "group": str(groups[index]),
                "identity_rmse": identity_error,
                "global_rmse": global_error,
                "self_fit_rmse": self_error,
                "self_fit_style_retention": self_style,
                "self_fit_dose": float(operators[index].dose),
                "group_oracle_case_id": ids[fit[oracle_position]],
                "group_oracle_rmse": float(case_errors[oracle_position]),
                "group_oracle_style_retention": float(case_style[oracle_position]),
                "style_eligible_case_count": int(len(eligible)),
                "style_constrained_case_id": (
                    ids[fit[constrained_position]]
                    if constrained_position is not None
                    else None
                ),
                "style_constrained_rmse": (
                    float(case_errors[constrained_position])
                    if constrained_position is not None
                    else None
                ),
                "style_constrained_style_retention": (
                    float(case_style[constrained_position])
                    if constrained_position is not None
                    else None
                ),
                "maximum_new_boundary_fraction": max(
                    _new_boundary_fraction(source, output, epsilon)
                    for output in selected_outputs
                ),
                "maximum_out_of_cube_fraction": max(
                    float(np.mean((output < 0.0) | (output > 1.0)))
                    for output in selected_outputs
                ),
            }
        )
    identity = np.asarray([row["identity_rmse"] for row in output_rows])
    global_error = np.asarray([row["global_rmse"] for row in output_rows])
    self_error = np.asarray([row["self_fit_rmse"] for row in output_rows])
    oracle_error = np.asarray([row["group_oracle_rmse"] for row in output_rows])
    available = np.asarray(
        [row["style_constrained_rmse"] is not None for row in output_rows]
    )
    constrained = np.asarray(
        [
            row["style_constrained_rmse"]
            for row in output_rows
            if row["style_constrained_rmse"] is not None
        ],
        dtype=np.float64,
    )
    comparable_oracle = np.asarray(
        [
            row["group_oracle_rmse"]
            for row in output_rows
            if row["style_constrained_rmse"] is not None
        ],
        dtype=np.float64,
    )
    constrained_ratio = (
        float(np.mean(constrained) / np.mean(comparable_oracle))
        if len(constrained)
        else None
    )
    all_diagnostics = diagnostics + [global_diagnostics]
    metrics = {
        "input_rows": len(all_rows),
        "fit_eligible_rows": len(ordered),
        "fit_ineligible_rows": len(all_rows) - len(ordered),
        "fit_eligible_fraction": eligible_fraction,
        "fit_ineligible_pair_ids": [
            str(row["pair_id"])
            for row, (eligible, _) in zip(all_rows, eligibility, strict=True)
            if not eligible
        ],
        "minimum_observed_strict_interior_fit_samples": min(
            count for _, count in eligibility
        ),
        "all_source_groups_retained": eligible_groups == all_groups,
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "fit_groups": sorted(set(map(str, groups[fit]))),
        "validation_groups": sorted(set(map(str, groups[validation]))),
        "coefficient_bank_sha256": _array_sha256(coefficients),
        "self_fit_mean_improvement_over_identity": float(
            (np.mean(identity) - np.mean(self_error)) / np.mean(identity)
        ),
        "self_fit_median_style_retention": float(
            np.median([row["self_fit_style_retention"] for row in output_rows])
        ),
        "group_oracle_mean_improvement_over_global": float(
            (np.mean(global_error) - np.mean(oracle_error)) / np.mean(global_error)
        ),
        "group_oracle_median_style_retention": float(
            np.median(
                [row["group_oracle_style_retention"] for row in output_rows]
            )
        ),
        "rows_with_style_eligible_fit_case_fraction": float(np.mean(available)),
        "style_constrained_oracle_error_ratio": constrained_ratio,
        "median_safe_dose": float(
            np.median([item[0].dose for item in operators_and_diagnostics])
        ),
        "minimum_safe_dose": float(
            min(item[0].dose for item in operators_and_diagnostics)
        ),
        "maximum_nonpositive_jacobian_count": int(
            max(int(item["nonpositive_determinant_count"]) for item in all_diagnostics)
        ),
        "maximum_new_boundary_fraction": float(
            max(row["maximum_new_boundary_fraction"] for row in output_rows)
        ),
        "maximum_out_of_cube_fraction": float(
            max(row["maximum_out_of_cube_fraction"] for row in output_rows)
        ),
    }
    checks = {
        "fit_eligibility": metrics["fit_eligible_fraction"]
        >= float(evaluation["minimum_fit_eligible_fraction"])
        and metrics["all_source_groups_retained"],
        "self_fit_accuracy": metrics["self_fit_mean_improvement_over_identity"]
        >= float(evaluation["minimum_self_fit_mean_improvement_over_identity"]),
        "self_fit_style": metrics["self_fit_median_style_retention"]
        >= float(evaluation["minimum_self_fit_median_style_retention"]),
        "group_oracle_value": metrics["group_oracle_mean_improvement_over_global"]
        >= float(evaluation["minimum_group_oracle_mean_improvement_over_global"]),
        "group_oracle_style": metrics["group_oracle_median_style_retention"]
        >= float(evaluation["minimum_group_oracle_median_style_retention"]),
        "style_eligible_availability": metrics[
            "rows_with_style_eligible_fit_case_fraction"
        ]
        >= float(evaluation["minimum_rows_with_style_eligible_fit_case"]),
        "constrained_error": constrained_ratio is not None
        and constrained_ratio
        <= float(evaluation["maximum_style_constrained_oracle_error_ratio"]),
        "median_dose": metrics["median_safe_dose"]
        >= float(evaluation["minimum_median_safe_dose"]),
        "minimum_dose": metrics["minimum_safe_dose"]
        >= float(evaluation["minimum_worst_safe_dose"]),
        "jacobian": metrics["maximum_nonpositive_jacobian_count"]
        <= int(evaluation["maximum_nonpositive_jacobian_count"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(evaluation["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"]
        <= float(evaluation["maximum_out_of_cube_fraction"]),
    }
    return {
        "metrics": metrics,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "rows": output_rows,
    }, coefficients


__all__ = [
    "FiveKProjectionCurveCasebankError",
    "evaluate_projection_curve_casebank",
    "fit_projection_curve_bank",
    "projection_curve_fit_eligible",
]
