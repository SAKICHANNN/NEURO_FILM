"""Style-capacity diagnosis for an already-fitted FiveK explicit case bank."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.boundary_safe_neutral_base import apply_boundary_safe_residual
from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_conditional_explicit_operator import effective_case_parameters
from src.eval.fivek_pairwise_compatibility import _group_partition
from src.eval.fivek_source_hard_retrieval import _rgb
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


class FiveKOperatorStyleCeilingError(ValueError):
    """Raised when the style-ceiling evidence contract drifts."""


def _safe_output(
    operator: TriangularLogitTransport,
    source: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    output, _ = apply_boundary_safe_residual(
        source[None, ...],
        operator.apply(source)[None, ...],
        boundary_epsilon=epsilon,
    )
    return output[0]


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


def diagnose_operator_style_ceiling(
    *,
    rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    split_spec: Mapping[str, Any],
    safe_spec: Mapping[str, Any],
    samples_per_image: int,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    bank = oracle_report["case_bank"]
    if ids != [str(row["pair_id"]) for row in bank]:
        raise FiveKOperatorStyleCeilingError("case identity drift")
    parameters = effective_case_parameters(oracle_report)
    operators = [TriangularLogitTransport(value) for value in parameters]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups,
        int(split_spec["group_bucket_modulus"]),
        int(split_spec["validation_bucket"]),
    )
    epsilon = float(safe_spec["boundary_epsilon"])
    threshold = float(gates["minimum_self_fit_median_style_retention"])
    output_rows: list[dict[str, Any]] = []
    for index in validation:
        source = _even_samples(_rgb(ordered[index]["source"]), samples_per_image)
        target = _even_samples(_rgb(ordered[index]["target"]), samples_per_image)
        target_style = _rmse(target, source)
        self_output = _safe_output(operators[index], source, epsilon)
        self_error = _rmse(self_output, target)
        self_retention = _rmse(self_output, source) / max(target_style, 1.0e-12)
        case_outputs = [
            _safe_output(operators[case], source, epsilon) for case in fit
        ]
        case_errors = np.asarray(
            [_rmse(output, target) for output in case_outputs], dtype=np.float64
        )
        case_retention = np.asarray(
            [
                _rmse(output, source) / max(target_style, 1.0e-12)
                for output in case_outputs
            ],
            dtype=np.float64,
        )
        error_position = int(np.argmin(case_errors))
        strongest_position = int(np.argmax(case_retention))
        eligible = np.flatnonzero(case_retention >= threshold)
        constrained_position = (
            int(eligible[np.argmin(case_errors[eligible])])
            if len(eligible)
            else None
        )
        candidate_outputs = [self_output, case_outputs[error_position]]
        if constrained_position is not None:
            candidate_outputs.append(case_outputs[constrained_position])
        output_rows.append(
            {
                "pair_id": ids[index],
                "group": str(groups[index]),
                "target_style_rmse": target_style,
                "self_fit_error": self_error,
                "self_fit_style_retention": self_retention,
                "error_oracle_case_id": ids[fit[error_position]],
                "error_oracle_error": float(case_errors[error_position]),
                "error_oracle_style_retention": float(
                    case_retention[error_position]
                ),
                "strongest_case_id": ids[fit[strongest_position]],
                "strongest_case_style_retention": float(
                    case_retention[strongest_position]
                ),
                "style_eligible_case_count": int(len(eligible)),
                "style_constrained_case_id": (
                    ids[fit[constrained_position]]
                    if constrained_position is not None
                    else None
                ),
                "style_constrained_error": (
                    float(case_errors[constrained_position])
                    if constrained_position is not None
                    else None
                ),
                "style_constrained_style_retention": (
                    float(case_retention[constrained_position])
                    if constrained_position is not None
                    else None
                ),
                "maximum_new_boundary_fraction": max(
                    _new_boundary_fraction(source, output, epsilon)
                    for output in candidate_outputs
                ),
            }
        )
    self_retention = np.asarray(
        [row["self_fit_style_retention"] for row in output_rows]
    )
    error_retention = np.asarray(
        [row["error_oracle_style_retention"] for row in output_rows]
    )
    available = np.asarray(
        [row["style_constrained_error"] is not None for row in output_rows]
    )
    constrained_errors = np.asarray(
        [
            row["style_constrained_error"]
            for row in output_rows
            if row["style_constrained_error"] is not None
        ],
        dtype=np.float64,
    )
    ordinary_errors = np.asarray(
        [
            row["error_oracle_error"]
            for row in output_rows
            if row["style_constrained_error"] is not None
        ],
        dtype=np.float64,
    )
    constrained_ratio = (
        float(np.mean(constrained_errors) / np.mean(ordinary_errors))
        if len(constrained_errors)
        else None
    )
    metrics = {
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "fit_groups": sorted(set(map(str, groups[fit]))),
        "validation_groups": sorted(set(map(str, groups[validation]))),
        "self_fit_median_style_retention": float(np.median(self_retention)),
        "error_oracle_median_style_retention": float(
            np.median(error_retention)
        ),
        "rows_with_style_eligible_fit_case_fraction": float(np.mean(available)),
        "style_constrained_oracle_error_ratio": constrained_ratio,
        "median_strongest_case_style_retention": float(
            np.median(
                [row["strongest_case_style_retention"] for row in output_rows]
            )
        ),
        "maximum_new_boundary_fraction": float(
            max(row["maximum_new_boundary_fraction"] for row in output_rows)
        ),
    }
    checks = {
        "self_fit_style": metrics["self_fit_median_style_retention"]
        >= float(gates["minimum_self_fit_median_style_retention"]),
        "error_oracle_style": metrics["error_oracle_median_style_retention"]
        >= float(gates["minimum_error_oracle_median_style_retention"]),
        "style_eligible_availability": metrics[
            "rows_with_style_eligible_fit_case_fraction"
        ]
        >= float(gates["minimum_rows_with_style_eligible_fit_case"]),
        "constrained_error": constrained_ratio is not None
        and constrained_ratio
        <= float(gates["maximum_style_constrained_oracle_error_ratio"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
    }
    if not checks["self_fit_style"]:
        branch = "operator_capacity_inadequate"
    elif not checks["error_oracle_style"]:
        branch = "selection_objective_loses_style"
    elif not checks["style_eligible_availability"]:
        branch = "style_eligible_case_bank_sparse"
    elif all(checks.values()):
        branch = "operator_capacity_adequate"
    else:
        branch = "style_constrained_oracle_not_viable"
    return {
        "metrics": metrics,
        "gates": checks,
        "branch": branch,
        "rows": output_rows,
    }


__all__ = [
    "FiveKOperatorStyleCeilingError",
    "diagnose_operator_style_ceiling",
]
