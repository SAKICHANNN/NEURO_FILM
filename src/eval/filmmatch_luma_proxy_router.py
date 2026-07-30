"""Held-frame hard router for the selective FilmMatch exposure expert."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_paired_source import canonical_sha256


def fit_conservative_threshold(
    feature: np.ndarray, label: np.ndarray
) -> tuple[float, dict[str, float]]:
    values = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(label, dtype=bool)
    if (
        values.ndim != 1
        or labels.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.any(labels)
        or np.all(labels)
    ):
        raise ValueError("threshold fit needs finite binary-class rows")
    unique = np.unique(values)
    candidates = [float(unique[0] - 1e-12)]
    candidates.extend(
        float((left + right) * 0.5)
        for left, right in zip(unique[:-1], unique[1:])
    )
    candidates.append(float(unique[-1] + 1e-12))
    rows = []
    for threshold in candidates:
        predicted = values >= threshold
        true_positive = int(np.count_nonzero(predicted & labels))
        false_positive = int(np.count_nonzero(predicted & ~labels))
        true_negative = int(np.count_nonzero(~predicted & ~labels))
        false_negative = int(np.count_nonzero(~predicted & labels))
        recall = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": 0.5 * (recall + specificity),
                "precision": precision,
                "recall": recall,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["precision"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), selected


def evaluate_luma_proxy_router(
    datasets: Mapping[str, Any],
    oracle_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    records = datasets["reflective_records"]
    group_ids = np.asarray(
        [
            f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}"
            for row in records
        ]
    )
    exposure = {
        f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}": int(
            row["exposure_ev"]
        )
        for row in records
    }
    groups = sorted(set(group_ids.tolist()))
    feature = {
        group: float(np.median(source[group_ids == group])) for group in groups
    }
    oracle_by_group = {
        str(row["held_group"]): row for row in oracle_report["folds"]
    }
    minimum_ev = int(config["router"]["high_exposure_minimum_ev"])
    rows = []
    for held_group in groups:
        development_groups = [
            group for group in groups if group != held_group
        ]
        development_feature = np.asarray(
            [feature[group] for group in development_groups]
        )
        development_label = np.asarray(
            [exposure[group] >= minimum_ev for group in development_groups]
        )
        threshold, fit = fit_conservative_threshold(
            development_feature, development_label
        )
        predicted_high = feature[held_group] >= threshold
        actual_high = exposure[held_group] >= minimum_ev
        oracle = oracle_by_group[held_group]
        global_rmse = float(oracle["metrics"]["global"]["rgb_rmse"])
        routed_rmse = float(
            oracle["metrics"][
                (
                    "correct_condition_expert"
                    if predicted_high
                    else "global"
                )
            ]["rgb_rmse"]
        )
        rows.append(
            {
                "held_group": held_group,
                "exposure_ev": exposure[held_group],
                "feature_median_rgb_code": feature[held_group],
                "development_threshold": threshold,
                "development_threshold_fit": fit,
                "actual_high_exposure": actual_high,
                "predicted_high_exposure": bool(predicted_high),
                "global_rgb_rmse": global_rmse,
                "routed_rgb_rmse": routed_rmse,
                "routed_improvement_over_global": float(
                    1.0 - routed_rmse / global_rmse
                ),
            }
        )
    actual = np.asarray([row["actual_high_exposure"] for row in rows])
    predicted = np.asarray([row["predicted_high_exposure"] for row in rows])
    true_positive = int(np.count_nonzero(actual & predicted))
    false_positive = int(np.count_nonzero(~actual & predicted))
    false_negative = int(np.count_nonzero(actual & ~predicted))
    true_negative = int(np.count_nonzero(~actual & ~predicted))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    improvements = np.asarray(
        [row["routed_improvement_over_global"] for row in rows]
    )
    routed_rows = improvements[predicted]
    aggregate = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "routed_folds": int(np.count_nonzero(predicted)),
        "routed_median_improvement": float(np.median(routed_rows)),
        "routed_worst_improvement": float(np.min(routed_rows)),
        "all_fold_mean_improvement": float(np.mean(improvements)),
        "all_fold_worst_improvement": float(np.min(improvements)),
    }
    gates = config["router_gate"]
    passed = bool(
        aggregate["precision"] >= float(gates["minimum_precision"])
        and aggregate["recall"] >= float(gates["minimum_recall"])
        and aggregate["balanced_accuracy"]
        >= float(gates["minimum_balanced_accuracy"])
        and aggregate["routed_median_improvement"]
        >= float(gates["minimum_routed_median_improvement"])
        and aggregate["routed_worst_improvement"]
        >= float(gates["minimum_routed_worst_improvement"])
        and aggregate["all_fold_mean_improvement"]
        >= float(gates["minimum_all_fold_mean_improvement"])
        and aggregate["all_fold_worst_improvement"]
        >= float(gates["minimum_all_fold_worst_improvement"])
    )
    report = {
        "schema": "neuro_film.u5_r2aw8_filmmatch_luma_router.v1",
        "experiment_id": config["experiment_id"],
        "folds": rows,
        "aggregate": aggregate,
        "router_gate_passed": passed,
        "validation_router_candidate_opened": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_luma_proxy_router", "fit_conservative_threshold"]
