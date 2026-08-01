"""Fit-only selection and held-group evaluation of one strong fixed LUT."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import _group_partition
from src.eval.fivek_strict_interior_lut_basis_development import apply_strict_interior_lut
from src.eval.fivek_strict_interior_lut_casebank import (
    _apply_bank,
    _new_boundary_fraction,
    fit_strict_interior_lut_bank,
)


class FiveKFixedStrongLUTChampionError(ValueError):
    """Raised when K=1 champion selection violates its frozen contract."""


def _apply(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return apply_strict_interior_lut(source[:, None, :], lut)[:, 0, :]


def evaluate_fixed_strong_lut_champion(
    *, rows: Sequence[Mapping[str, Any]], operator: Mapping[str, Any],
    split_spec: Mapping[str, Any], selection: Mapping[str, Any],
    evaluation: Mapping[str, Any], samples_per_image: int,
) -> tuple[dict[str, Any], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(groups, int(split_spec["group_bucket_modulus"]), int(split_spec["validation_bucket"]))
    coefficients = fit_strict_interior_lut_bank(ordered, operator)
    sampled = [
        (
            _even_samples(np.asarray(row["source"]), samples_per_image),
            _even_samples(np.asarray(row["target"]), samples_per_image),
        )
        for row in ordered
    ]
    error_matrix = np.empty((len(fit), len(fit)), dtype=np.float64)
    style_matrix = np.empty_like(error_matrix)
    for query_position, query_index in enumerate(fit):
        source, target = sampled[query_index]
        outputs = _apply_bank(source, coefficients[fit])
        identity_error = _rmse(source, target)
        error_matrix[query_position] = np.sqrt(np.mean((outputs - target[None, ...]) ** 2, axis=(1, 2)))
        style_matrix[query_position] = np.sqrt(np.mean((outputs - source[None, ...]) ** 2, axis=(1, 2))) / max(identity_error, 1e-12)
    style_floor = float(selection["minimum_cross_group_median_target_style_retention"])
    fraction_floor = float(selection["minimum_cross_group_style_eligible_fraction"])
    candidates = []
    for case_position, case_index in enumerate(fit):
        eligible_rows = groups[fit] != groups[case_index]
        errors = error_matrix[eligible_rows, case_position]
        styles = style_matrix[eligible_rows, case_position]
        record = {
            "case_id": ids[case_index],
            "case_index": int(case_index),
            "cross_group_rows": int(np.sum(eligible_rows)),
            "mean_rmse": float(np.mean(errors)),
            "median_target_style_retention": float(np.median(styles)),
            "style_eligible_fraction": float(np.mean(styles >= style_floor)),
        }
        record["eligible"] = (
            record["median_target_style_retention"] >= style_floor
            and record["style_eligible_fraction"] >= fraction_floor
        )
        candidates.append(record)
    eligible_candidates = [record for record in candidates if record["eligible"]]
    if not eligible_candidates:
        raise FiveKFixedStrongLUTChampionError("no fit-only strong champion candidate")
    champion = min(eligible_candidates, key=lambda record: (record["mean_rmse"], record["case_id"]))
    champion_lut = coefficients[int(champion["case_index"])]
    global_lut = np.median(coefficients[fit], axis=0)
    doses = [float(value) for value in evaluation["strength_doses"]]
    records = []
    global_errors, champion_errors, strength_errors = [], [], []
    champion_styles, boundaries, cube = [], [], []
    for index in validation:
        source, target = sampled[index]
        identity_error = _rmse(source, target)
        global_output = _apply(source, global_lut)
        champion_output = _apply(source, champion_lut)
        global_error = _rmse(global_output, target)
        champion_error = _rmse(champion_output, target)
        strength_error = min(_rmse(_apply(source, global_lut * dose), target) for dose in doses)
        style = _rmse(champion_output, source) / max(identity_error, 1e-12)
        global_errors.append(global_error); champion_errors.append(champion_error); strength_errors.append(strength_error); champion_styles.append(style)
        boundaries.append(max(_new_boundary_fraction(source, output, float(operator["boundary_epsilon"])) for output in (global_output, champion_output)))
        cube.append(max(float(np.mean((output < 0.0) | (output > 1.0))) for output in (global_output, champion_output)))
        records.append({
            "pair_id": ids[index], "group": str(groups[index]),
            "champion_case_id": champion["case_id"], "global_rmse": global_error,
            "champion_rmse": champion_error, "global_strength_oracle_rmse": strength_error,
            "champion_target_style_retention": style,
        })
    global_array = np.asarray(global_errors); champion_array = np.asarray(champion_errors); strength_array = np.asarray(strength_errors)
    metrics = {
        "fit_rows": len(fit), "validation_rows": len(validation),
        "eligible_candidate_count": len(eligible_candidates),
        "selected_champion": {key: value for key, value in champion.items() if key != "case_index"},
        "mean_error_ratio_to_coefficient_median_global": float(np.mean(champion_array) / np.mean(global_array)),
        "p95_error_ratio_to_coefficient_median_global": float(np.quantile(champion_array, 0.95) / np.quantile(global_array, 0.95)),
        "worst_error_ratio_to_coefficient_median_global": float(np.max(champion_array) / np.max(global_array)),
        "mean_error_ratio_to_global_strength_oracle": float(np.mean(champion_array) / np.mean(strength_array)),
        "win_fraction_over_coefficient_median_global": float(np.mean(champion_array < global_array)),
        "median_target_style_retention": float(np.median(champion_styles)),
        "style_eligible_fraction": float(np.mean(np.asarray(champion_styles) >= float(evaluation["minimum_median_target_style_retention"]))),
        "maximum_new_boundary_fraction": float(max(boundaries)),
        "maximum_out_of_cube_fraction": float(max(cube)),
    }
    gates = {
        "mean": metrics["mean_error_ratio_to_coefficient_median_global"] <= float(evaluation["maximum_mean_error_ratio_to_coefficient_median_global"]),
        "p95": metrics["p95_error_ratio_to_coefficient_median_global"] <= float(evaluation["maximum_p95_error_ratio_to_coefficient_median_global"]),
        "worst": metrics["worst_error_ratio_to_coefficient_median_global"] <= float(evaluation["maximum_worst_error_ratio_to_coefficient_median_global"]),
        "strength": metrics["mean_error_ratio_to_global_strength_oracle"] <= float(evaluation["maximum_mean_error_ratio_to_global_strength_oracle"]),
        "style": metrics["median_target_style_retention"] >= float(evaluation["minimum_median_target_style_retention"]),
        "style_coverage": metrics["style_eligible_fraction"] >= float(evaluation["minimum_style_eligible_fraction"]),
        "boundary": metrics["maximum_new_boundary_fraction"] <= float(evaluation["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"] <= float(evaluation["maximum_out_of_cube_fraction"]),
    }
    return {"metrics": metrics, "gates": gates, "automatic_pass": all(gates.values()), "rows": records}, coefficients


__all__ = ["FiveKFixedStrongLUTChampionError", "evaluate_fixed_strong_lut_champion"]
