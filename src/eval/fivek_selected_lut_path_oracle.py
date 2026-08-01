"""Evaluator-only strength Oracle on fixed source-selected LUT directions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _rmse
from src.eval.fivek_pairwise_compatibility import _group_partition
from src.eval.fivek_strict_interior_lut_basis_development import apply_strict_interior_lut
from src.eval.fivek_strict_interior_lut_casebank import _new_boundary_fraction


class FiveKSelectedLUTPathOracleError(ValueError):
    """Raised when fixed selected-path evidence is inconsistent."""


def _apply(source: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return apply_strict_interior_lut(source[:, None, :], lut)[:, 0, :]


def evaluate_selected_lut_path_oracle(
    *, rows: Sequence[Mapping[str, Any]], coefficients: np.ndarray,
    parent_variant: Mapping[str, Any], split_spec: Mapping[str, Any],
    path_spec: Mapping[str, Any], evaluation: Mapping[str, Any],
    samples_per_image: int,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in ordered]
    id_to_index = {pair_id: index for index, pair_id in enumerate(ids)}
    groups = np.asarray([str(row["group"]) for row in ordered], dtype=object)
    fit, validation = _group_partition(
        groups, int(split_spec["group_bucket_modulus"]), int(split_spec["validation_bucket"])
    )
    parent_rows = {str(row["pair_id"]): row for row in parent_variant["rows"]}
    if set(parent_rows) != {ids[index] for index in validation}:
        raise FiveKSelectedLUTPathOracleError("parent selected-row identity drift")
    luts = np.asarray(coefficients, dtype=np.float64)
    if luts.shape != (len(ordered), 4, 4, 4, 3):
        raise FiveKSelectedLUTPathOracleError("coefficient bank shape drift")
    global_lut = np.median(luts[fit], axis=0)
    doses = [float(value) for value in path_spec["doses"]]
    if doses != sorted(set(doses)) or doses[0] != 0.0 or doses[-1] != 1.0:
        raise FiveKSelectedLUTPathOracleError("invalid path doses")
    style_floor = float(path_spec["minimum_target_style_retention"])
    records = []
    arrays = {name: [] for name in ("global", "unconstrained", "constrained", "strength")}
    availability = []
    boundary = []
    cube = []
    constrained_styles = []
    for index in validation:
        row = ordered[index]
        source = _even_samples(np.asarray(row["source"]), samples_per_image)
        target = _even_samples(np.asarray(row["target"]), samples_per_image)
        identity_error = _rmse(source, target)
        parent = parent_rows[ids[index]]
        selected_index = id_to_index[str(parent["selected_case_id"])]
        case_lut = luts[selected_index]
        path_outputs = [
            _apply(source, global_lut + dose * (case_lut - global_lut))
            for dose in doses
        ]
        errors = np.asarray([_rmse(output, target) for output in path_outputs])
        styles = np.asarray([
            _rmse(output, source) / max(identity_error, 1e-12)
            for output in path_outputs
        ])
        unconstrained_position = int(np.argmin(errors))
        eligible = np.flatnonzero(styles >= style_floor)
        constrained_position = int(eligible[np.argmin(errors[eligible])]) if len(eligible) else None
        global_error = float(errors[0])
        strength_errors = [
            _rmse(_apply(source, global_lut * dose), target) for dose in doses
        ]
        constrained_error = (
            float(errors[constrained_position]) if constrained_position is not None else global_error
        )
        constrained_style = (
            float(styles[constrained_position]) if constrained_position is not None else float(styles[0])
        )
        arrays["global"].append(global_error)
        arrays["unconstrained"].append(float(errors[unconstrained_position]))
        arrays["constrained"].append(constrained_error)
        arrays["strength"].append(min(strength_errors))
        availability.append(constrained_position is not None)
        constrained_styles.append(constrained_style)
        selected_outputs = [path_outputs[unconstrained_position]]
        if constrained_position is not None:
            selected_outputs.append(path_outputs[constrained_position])
        boundary.append(max(_new_boundary_fraction(source, output, 1.0 / 510.0) for output in selected_outputs))
        cube.append(max(float(np.mean((output < 0.0) | (output > 1.0))) for output in selected_outputs))
        records.append({
            "pair_id": ids[index], "group": str(groups[index]),
            "selected_case_id": ids[selected_index], "parent_fallback": bool(parent["fallback"]),
            "global_rmse": global_error,
            "unconstrained_path_rmse": float(errors[unconstrained_position]),
            "unconstrained_dose": doses[unconstrained_position],
            "constrained_path_rmse": constrained_error,
            "constrained_dose": doses[constrained_position] if constrained_position is not None else None,
            "constrained_target_style_retention": constrained_style,
            "style_constrained_available": constrained_position is not None,
            "global_strength_oracle_rmse": min(strength_errors),
        })
    values = {name: np.asarray(items, dtype=np.float64) for name, items in arrays.items()}
    constrained_mean = float(np.mean(values["constrained"]))
    global_mean = float(np.mean(values["global"]))
    strength_mean = float(np.mean(values["strength"]))
    metrics = {
        "fit_rows": len(fit), "validation_rows": len(validation),
        "mean_improvement_over_global": (global_mean - constrained_mean) / max(global_mean, 1e-12),
        "win_fraction_over_global": float(np.mean(values["constrained"] < values["global"])),
        "p95_ratio_to_global": float(np.quantile(values["constrained"], 0.95) / max(np.quantile(values["global"], 0.95), 1e-12)),
        "worst_ratio_to_global": float(np.max(values["constrained"]) / max(np.max(values["global"]), 1e-12)),
        "mean_improvement_over_global_strength_oracle": (strength_mean - constrained_mean) / max(strength_mean, 1e-12),
        "win_fraction_over_global_strength_oracle": float(np.mean(values["constrained"] < values["strength"])),
        "style_constrained_availability": float(np.mean(availability)),
        "style_constraint_error_ratio": float(np.mean(values["constrained"]) / max(np.mean(values["unconstrained"]), 1e-12)),
        "median_target_style_retention": float(np.median(constrained_styles)),
        "median_constrained_dose": float(np.median([row["constrained_dose"] for row in records if row["constrained_dose"] is not None])),
        "maximum_new_boundary_fraction": float(max(boundary)),
        "maximum_out_of_cube_fraction": float(max(cube)),
    }
    gates = {
        "mean": metrics["mean_improvement_over_global"] >= float(evaluation["minimum_mean_improvement_over_global"]),
        "wins": metrics["win_fraction_over_global"] >= float(evaluation["minimum_win_fraction_over_global"]),
        "p95": metrics["p95_ratio_to_global"] <= float(evaluation["maximum_p95_ratio_to_global"]),
        "worst": metrics["worst_ratio_to_global"] <= float(evaluation["maximum_worst_ratio_to_global"]),
        "strength_mean": metrics["mean_improvement_over_global_strength_oracle"] >= float(evaluation["minimum_mean_improvement_over_global_strength_oracle"]),
        "strength_wins": metrics["win_fraction_over_global_strength_oracle"] >= float(evaluation["minimum_win_fraction_over_global_strength_oracle"]),
        "availability": metrics["style_constrained_availability"] >= float(evaluation["minimum_style_constrained_availability"]),
        "constraint_cost": metrics["style_constraint_error_ratio"] <= float(evaluation["maximum_style_constraint_error_ratio"]),
        "style": metrics["median_target_style_retention"] >= float(evaluation["minimum_median_target_style_retention"]),
        "boundary": metrics["maximum_new_boundary_fraction"] <= float(evaluation["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"] <= float(evaluation["maximum_out_of_cube_fraction"]),
    }
    return {"metrics": metrics, "gates": gates, "automatic_pass": all(gates.values()), "rows": records}


__all__ = ["FiveKSelectedLUTPathOracleError", "evaluate_selected_lut_path_oracle"]
