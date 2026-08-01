"""Mature strict-interior LUT capacity baseline on the large FiveK case bank."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_adaptive_lut_basis_development import _trilinear_features
from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import _group_partition
from src.eval.fivek_source_hard_retrieval import _rgb
from src.eval.fivek_strict_interior_lut_basis_development import (
    EPSILON,
    HEADROOM_SAFETY_FACTOR,
    apply_strict_interior_lut,
    fit_strict_interior_lut,
)


class FiveKStrictInteriorLUTCasebankError(ValueError):
    """Raised when the mature strict-interior LUT baseline drifts."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


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


def fit_strict_interior_lut_bank(
    rows: Sequence[Mapping[str, Any]], operator: Mapping[str, Any]
) -> np.ndarray:
    coefficients = [
        fit_strict_interior_lut(
            _rgb(row["source"]),
            _rgb(row["target"]),
            grid_size=int(operator["grid_size"]),
            sample_stride=int(operator["fit_sample_stride"]),
            identity_shrinkage=float(operator["identity_shrinkage"]),
            smoothness=float(operator["first_difference_smoothness"]),
            coefficient_minimum=float(operator["node_coefficient_minimum"]),
            coefficient_maximum=float(operator["node_coefficient_maximum"]),
        )
        for row in rows
    ]
    result = np.stack(coefficients)
    if (
        not np.all(np.isfinite(result))
        or np.any(result < -1.0)
        or np.any(result > 1.0)
    ):
        raise FiveKStrictInteriorLUTCasebankError("invalid coefficient bank")
    return result


def _apply_bank(source: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply every LUT while computing source interpolation features once."""

    source_array = np.asarray(source, dtype=np.float64)
    features = _trilinear_features(source_array, 4)
    headroom = HEADROOM_SAFETY_FACTOR * np.maximum(
        0.0, np.minimum(source_array - EPSILON, 1.0 - EPSILON - source_array)
    )
    residual = np.einsum(
        "...f,kfc->k...c",
        features,
        np.asarray(coefficients, dtype=np.float64).reshape(-1, 64, 3),
        optimize=True,
    )
    output = source_array[None, ...] + headroom[None, ...] * residual
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise FiveKStrictInteriorLUTCasebankError(
            "strict-interior case-bank proof failed"
        )
    return output


def evaluate_strict_interior_lut_casebank(
    *,
    rows: Sequence[Mapping[str, Any]],
    operator: Mapping[str, Any],
    split_spec: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    samples_per_image: int,
) -> tuple[dict[str, Any], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if not ordered:
        raise FiveKStrictInteriorLUTCasebankError("empty case bank")
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups,
        int(split_spec["group_bucket_modulus"]),
        int(split_spec["validation_bucket"]),
    )
    coefficients = fit_strict_interior_lut_bank(ordered, operator)
    global_coefficients = np.median(coefficients[fit], axis=0)
    epsilon = float(operator["boundary_epsilon"])
    style_threshold = float(
        evaluation["minimum_self_fit_median_style_retention"]
    )
    output_rows: list[dict[str, Any]] = []
    for index in validation:
        source = _even_samples(_rgb(ordered[index]["source"]), samples_per_image)
        target = _even_samples(_rgb(ordered[index]["target"]), samples_per_image)
        identity_error = _rmse(source, target)
        global_output = apply_strict_interior_lut(
            source[:, None, :], global_coefficients
        )[:, 0, :]
        self_output = apply_strict_interior_lut(
            source[:, None, :], coefficients[index]
        )[:, 0, :]
        case_outputs = _apply_bank(source, coefficients[fit])
        case_errors = np.sqrt(
            np.mean((case_outputs - target[None, ...]) ** 2, axis=(1, 2))
        )
        case_style = np.sqrt(
            np.mean((case_outputs - source[None, ...]) ** 2, axis=(1, 2))
        ) / max(identity_error, 1.0e-12)
        oracle_position = int(np.argmin(case_errors))
        eligible = np.flatnonzero(case_style >= style_threshold)
        constrained_position = (
            int(eligible[np.argmin(case_errors[eligible])])
            if len(eligible)
            else None
        )
        selected = [global_output, self_output, case_outputs[oracle_position]]
        if constrained_position is not None:
            selected.append(case_outputs[constrained_position])
        selected_array = np.stack(selected)
        output_rows.append(
            {
                "pair_id": ids[index],
                "group": str(groups[index]),
                "identity_rmse": identity_error,
                "global_rmse": _rmse(global_output, target),
                "self_fit_rmse": _rmse(self_output, target),
                "self_fit_style_retention": _rmse(self_output, source)
                / max(identity_error, 1.0e-12),
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
                "maximum_new_boundary_fraction": _new_boundary_fraction(
                    source[None, ...], selected_array, epsilon
                ),
                "maximum_out_of_cube_fraction": float(
                    np.mean((selected_array < 0.0) | (selected_array > 1.0))
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
    metrics = {
        "input_rows": len(ordered),
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
            np.median([row["group_oracle_style_retention"] for row in output_rows])
        ),
        "rows_with_style_eligible_fit_case_fraction": float(np.mean(available)),
        "style_constrained_oracle_error_ratio": constrained_ratio,
        "maximum_new_boundary_fraction": float(
            max(row["maximum_new_boundary_fraction"] for row in output_rows)
        ),
        "maximum_out_of_cube_fraction": float(
            max(row["maximum_out_of_cube_fraction"] for row in output_rows)
        ),
    }
    checks = {
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
    "FiveKStrictInteriorLUTCasebankError",
    "evaluate_strict_interior_lut_casebank",
    "fit_strict_interior_lut_bank",
]
