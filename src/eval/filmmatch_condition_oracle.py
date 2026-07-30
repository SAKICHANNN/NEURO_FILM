"""Condition-specific operator Oracle for the FilmMatch controlled session."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.monotone_curve_matrix import (
    fit_monotone_curve_positive_matrix,
)


def _fit(
    source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
) -> Any:
    fit = config["candidate"]["fit"]
    return fit_monotone_curve_positive_matrix(
        source,
        target,
        curve_identity_mixture=float(fit["curve_identity_mixture"]),
        matrix_identity_mixture=float(fit["matrix_identity_mixture"]),
        free_logit_bounds=tuple(map(float, fit["free_logit_bounds"])),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(fit["maximum_function_evaluations"]),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    ).operator


def _held_group_oracle(
    source: np.ndarray,
    target: np.ndarray,
    condition: np.ndarray,
    groups: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for held_group in sorted(set(groups.tolist())):
        held = groups == held_group
        held_condition = str(condition[np.flatnonzero(held)[0]])
        development = ~held
        same_condition = development & (condition == held_condition)
        if np.count_nonzero(same_condition) < 12:
            raise ValueError("condition expert lacks development support")
        global_operator = _fit(source[development], target[development], config)
        expert_operator = _fit(
            source[same_condition], target[same_condition], config
        )
        metrics = {
            "global": prediction_metrics(
                global_operator.apply(source[held]), target[held]
            ),
            "correct_condition_expert": prediction_metrics(
                expert_operator.apply(source[held]), target[held]
            ),
        }
        rows.append(
            {
                "held_group": str(held_group),
                "condition": held_condition,
                "held_samples": int(np.count_nonzero(held)),
                "same_condition_development_samples": int(
                    np.count_nonzero(same_condition)
                ),
                "metrics": metrics,
                "expert_improvement_over_global": float(
                    1.0
                    - metrics["correct_condition_expert"]["rgb_rmse"]
                    / metrics["global"]["rgb_rmse"]
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = np.asarray(
        [row["expert_improvement_over_global"] for row in rows]
    )
    return {
        "folds": len(rows),
        "expert_wins": int(np.count_nonzero(improvements > 0.0)),
        "expert_win_fraction": float(np.mean(improvements > 0.0)),
        "median_improvement_over_global": float(np.median(improvements)),
        "mean_improvement_over_global": float(np.mean(improvements)),
        "worst_improvement_over_global": float(np.min(improvements)),
        "median_global_rgb_rmse": float(
            np.median(
                [row["metrics"]["global"]["rgb_rmse"] for row in rows]
            )
        ),
        "median_expert_rgb_rmse": float(
            np.median(
                [
                    row["metrics"]["correct_condition_expert"]["rgb_rmse"]
                    for row in rows
                ]
            )
        ),
    }


def evaluate_condition_oracle(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rs = np.asarray(datasets["reflective_source"], dtype=np.float64)
    rt = np.asarray(datasets["reflective_target"], dtype=np.float64)
    rr = datasets["reflective_records"]
    reflective_condition = np.asarray([row["illuminant"] for row in rr])
    reflective_group = np.asarray(
        [
            f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}"
            for row in rr
        ]
    )
    es = np.asarray(datasets["emissive_source"], dtype=np.float64)
    et = np.asarray(datasets["emissive_target"], dtype=np.float64)
    er = datasets["emissive_records"]
    emissive_condition = np.asarray([row["hue_sector"] for row in er])
    emissive_group = np.asarray(
        [f"stimulus={int(row['stimulus_index']):02d}" for row in er]
    )
    reflective_rows = _held_group_oracle(
        rs,
        rt,
        reflective_condition,
        reflective_group,
        config=config,
    )
    emissive_rows = _held_group_oracle(
        es,
        et,
        emissive_condition,
        emissive_group,
        config=config,
    )
    reflective = _aggregate(reflective_rows)
    emissive = _aggregate(emissive_rows)
    gates = config["oracle_gate"]
    passed = bool(
        reflective["median_improvement_over_global"]
        >= float(gates["minimum_reflective_median_improvement"])
        and reflective["expert_win_fraction"]
        >= float(gates["minimum_reflective_win_fraction"])
        and emissive["median_improvement_over_global"]
        >= float(gates["minimum_emissive_median_improvement"])
        and emissive["expert_win_fraction"]
        >= float(gates["minimum_emissive_win_fraction"])
    )
    report = {
        "schema": "neuro_film.u5_r2aw4_filmmatch_condition_oracle.v1",
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_rows,
        "emissive_folds": emissive_rows,
        "reflective_aggregate": reflective,
        "emissive_aggregate": emissive,
        "oracle_gate_passed": passed,
        "conditional_router_research_opened": passed,
        "promotion_opened": False,
        "interpretation": (
            "controlled condition expert value, not a film mode or stock signal"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


def evaluate_exposure_oracle(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure a known-EV expert without claiming EV inference."""

    source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = datasets["reflective_records"]
    condition = np.asarray(
        [f"ev={int(row['exposure_ev']):+d}" for row in records]
    )
    groups = np.asarray(
        [
            f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}"
            for row in records
        ]
    )
    rows = _held_group_oracle(
        source,
        target,
        condition,
        groups,
        config=config,
    )
    aggregate = _aggregate(rows)
    gates = config["oracle_gate"]
    passed = bool(
        aggregate["median_improvement_over_global"]
        >= float(gates["minimum_median_improvement"])
        and aggregate["expert_win_fraction"]
        >= float(gates["minimum_win_fraction"])
        and aggregate["worst_improvement_over_global"]
        >= float(gates["minimum_worst_improvement"])
    )
    report = {
        "schema": "neuro_film.u5_r2aw5_filmmatch_exposure_oracle.v1",
        "experiment_id": config["experiment_id"],
        "folds": rows,
        "aggregate": aggregate,
        "oracle_gate_passed": passed,
        "exposure_proxy_retrieval_research_opened": passed,
        "promotion_opened": False,
        "interpretation": (
            "known controlled exposure-condition evaluator Oracle only; "
            "no exposure inference or physical film mode"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_condition_oracle", "evaluate_exposure_oracle"]
