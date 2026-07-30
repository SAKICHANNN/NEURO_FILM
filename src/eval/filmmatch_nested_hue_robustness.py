"""Nested leave-one-hue-out robustness for domain-loss selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_domain_balanced_capacity import (
    _domain_weights,
    _fit,
)
from src.eval.filmmatch_paired_source import canonical_sha256


def evaluate_nested_hue_robustness(
    datasets: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    rs = np.asarray(datasets["reflective_source"], dtype=np.float64)
    rt = np.asarray(datasets["reflective_target"], dtype=np.float64)
    es = np.asarray(datasets["emissive_source"], dtype=np.float64)
    et = np.asarray(datasets["emissive_target"], dtype=np.float64)
    hues = np.asarray(
        [row["hue_sector"] for row in datasets["emissive_records"]]
    )
    hue_names = sorted(set(hues))
    variants = list(config["candidate"]["variants"])
    outer_rows = []
    for outer_hue in hue_names:
        remaining_hues = [hue for hue in hue_names if hue != outer_hue]
        variant_inner = {}
        for variant in variants:
            inner_rows = []
            for inner_hue in remaining_hues:
                train_mask = (hues != outer_hue) & (hues != inner_hue)
                weights = _domain_weights(
                    len(rs),
                    int(np.sum(train_mask)),
                    float(variant["emissive_loss_share"]),
                )
                fit = _fit(
                    np.concatenate([rs, es[train_mask]]),
                    np.concatenate([rt, et[train_mask]]),
                    variant,
                    config,
                    sample_weights=weights,
                )
                metrics = prediction_metrics(
                    fit.operator.apply(es[hues == inner_hue]),
                    et[hues == inner_hue],
                )
                inner_rows.append(
                    {
                        "held_inner_hue": inner_hue,
                        "rgb_rmse": metrics["rgb_rmse"],
                        "safe_residual_strength": fit.operator.residual.strength,
                    }
                )
            variant_inner[str(variant["id"])] = {
                "mean_rgb_rmse": float(
                    np.mean([row["rgb_rmse"] for row in inner_rows])
                ),
                "folds": inner_rows,
            }
        selected_id = min(
            (str(variant["id"]) for variant in variants),
            key=lambda name: (
                variant_inner[name]["mean_rgb_rmse"],
                next(
                    index
                    for index, variant in enumerate(variants)
                    if str(variant["id"]) == name
                ),
            ),
        )
        selected = next(
            variant for variant in variants if str(variant["id"]) == selected_id
        )
        outer_train = hues != outer_hue
        outer_weights = _domain_weights(
            len(rs),
            int(np.sum(outer_train)),
            float(selected["emissive_loss_share"]),
        )
        outer_fit = _fit(
            np.concatenate([rs, es[outer_train]]),
            np.concatenate([rt, et[outer_train]]),
            selected,
            config,
            sample_weights=outer_weights,
        )
        outer_metrics = prediction_metrics(
            outer_fit.operator.apply(es[hues == outer_hue]),
            et[hues == outer_hue],
        )
        identity_metrics = prediction_metrics(
            es[hues == outer_hue], et[hues == outer_hue]
        )
        improvement = float(
            1.0 - outer_metrics["rgb_rmse"] / identity_metrics["rgb_rmse"]
        )
        outer_rows.append(
            {
                "held_outer_hue": outer_hue,
                "inner_selection": variant_inner,
                "selected_variant": selected_id,
                "selected_emissive_loss_share": float(
                    selected["emissive_loss_share"]
                ),
                "outer_metrics": outer_metrics,
                "outer_identity_rgb_rmse": identity_metrics["rgb_rmse"],
                "outer_improvement_over_identity": improvement,
                "safe_residual_strength": outer_fit.operator.residual.strength,
            }
        )
    improvements = np.asarray(
        [row["outer_improvement_over_identity"] for row in outer_rows]
    )
    aggregate = {
        "outer_fold_count": len(outer_rows),
        "median_improvement_over_identity": float(np.median(improvements)),
        "mean_improvement_over_identity": float(np.mean(improvements)),
        "worst_improvement_over_identity": float(np.min(improvements)),
        "win_count": int(np.sum(improvements > 0.0)),
        "selected_variant_counts": {
            str(variant["id"]): sum(
                row["selected_variant"] == str(variant["id"])
                for row in outer_rows
            )
            for variant in variants
        },
    }
    gates = config["development_readout"]
    passed = bool(
        aggregate["median_improvement_over_identity"]
        >= float(gates["minimum_median_improvement_over_identity"])
        and aggregate["win_count"] >= int(gates["minimum_outer_win_count"])
        and aggregate["worst_improvement_over_identity"]
        >= float(gates["minimum_worst_improvement_over_identity"])
    )
    report = {
        "schema": "neuro_film.u5_r2ax8_filmmatch_nested_hue_robustness.v1",
        "experiment_id": config["experiment_id"],
        "outer_folds": outer_rows,
        "aggregate": aggregate,
        "development_readout_passed": passed,
        "independent_confirmation_opened": False,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_nested_hue_robustness"]
