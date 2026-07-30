"""Grouped FilmMatch capacity for triangular conditional monotone maps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.triangular_conditional_monotone import (
    fit_triangular_conditional_monotone,
)


def _fit(
    source: np.ndarray,
    target: np.ndarray,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    fit = config["candidate"]["fit"]
    return fit_triangular_conditional_monotone(
        source,
        target,
        channel_order=tuple(int(value) for value in variant["channel_order"]),
        segment_count=int(variant["segment_count"]),
        learned_mixture=float(variant["learned_mixture"]),
        free_logit_bounds=tuple(map(float, fit["free_logit_bounds"])),
        conditioner_l2=float(fit["conditioner_l2"]),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(fit["maximum_function_evaluations"]),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    )


def _fold(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    held_source: np.ndarray,
    held_target: np.ndarray,
    *,
    fold_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {}
    for variant in config["candidate"]["variants"]:
        fit = _fit(fit_source, fit_target, variant, config)
        rows[str(variant["id"])] = {
            "metrics": prediction_metrics(
                fit.operator.apply(held_source), held_target
            ),
            "converged": fit.converged,
            "development_rgb_rmse": fit.development_rgb_rmse,
        }
    return {"fold_id": fold_id, "variants": rows}


def _aggregate(folds: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        rmse = np.asarray(
            [row["variants"][name]["metrics"]["rgb_rmse"] for row in folds]
        )
        result[name] = {
            "median_rgb_rmse": float(np.median(rmse)),
            "mean_rgb_rmse": float(np.mean(rmse)),
            "worst_rgb_rmse": float(np.max(rmse)),
            "all_converged": bool(
                all(row["variants"][name]["converged"] for row in folds)
            ),
        }
    return result


def evaluate_triangular_conditional_capacity(
    datasets: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    rs = np.asarray(datasets["reflective_source"], dtype=np.float64)
    rt = np.asarray(datasets["reflective_target"], dtype=np.float64)
    es = np.asarray(datasets["emissive_source"], dtype=np.float64)
    et = np.asarray(datasets["emissive_target"], dtype=np.float64)
    illuminants = np.asarray(
        [row["illuminant"] for row in datasets["reflective_records"]]
    )
    hues = np.asarray(
        [row["hue_sector"] for row in datasets["emissive_records"]]
    )
    reflective_folds = [
        _fold(
            rs[illuminants != name],
            rt[illuminants != name],
            rs[illuminants == name],
            rt[illuminants == name],
            fold_id=f"held-illuminant-{name}",
            config=config,
        )
        for name in ("5600K", "3200K", "5600K_CTB")
    ]
    hue_folds = [
        _fold(
            np.concatenate([rs, es[hues != hue]]),
            np.concatenate([rt, et[hues != hue]]),
            es[hues == hue],
            et[hues == hue],
            fold_id=f"held-emissive-hue-{hue}",
            config=config,
        )
        for hue in sorted(set(hues))
    ]
    names = [str(row["id"]) for row in config["candidate"]["variants"]]
    reflective = _aggregate(reflective_folds, names)
    emissive = _aggregate(hue_folds, names)
    parent = config["parent_capacity"]
    for name in names:
        reflective[name]["improvement_over_identity"] = float(
            1.0
            - reflective[name]["median_rgb_rmse"]
            / float(parent["reflective_identity_median_rmse"])
        )
        reflective[name]["improvement_over_affine"] = float(
            1.0
            - reflective[name]["median_rgb_rmse"]
            / float(parent["reflective_affine_median_rmse"])
        )
        emissive[name]["improvement_over_identity"] = float(
            1.0
            - emissive[name]["median_rgb_rmse"]
            / float(parent["emissive_identity_median_rmse"])
        )
    score = {
        name: reflective[name]["mean_rgb_rmse"] + emissive[name]["mean_rgb_rmse"]
        for name in names
    }
    selected = min(names, key=lambda name: (score[name], names.index(name)))
    gates = config["development_readout"]
    passed = bool(
        reflective[selected]["improvement_over_identity"]
        >= float(gates["minimum_reflective_improvement_over_identity"])
        and reflective[selected]["improvement_over_affine"]
        >= float(gates["minimum_reflective_improvement_over_affine"])
        and emissive[selected]["improvement_over_identity"]
        >= float(gates["minimum_emissive_improvement_over_identity"])
        and reflective[selected]["all_converged"]
        and emissive[selected]["all_converged"]
    )
    variant = next(
        row for row in config["candidate"]["variants"] if row["id"] == selected
    )
    final = _fit(
        np.concatenate([rs, es]), np.concatenate([rt, et]), variant, config
    )
    axis = np.linspace(0.01, 0.99, 9)
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    report = {
        "schema": "neuro_film.u5_r2ax2_filmmatch_triangular_conditional.v1",
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_folds,
        "emissive_hue_folds": hue_folds,
        "reflective_aggregate": reflective,
        "emissive_hue_aggregate": emissive,
        "selection": {"selected": selected, "score": score},
        "final_fit": {
            "operator": final.operator.to_dict(),
            "development_rgb_rmse": final.development_rgb_rmse,
            "minimum_sampled_jacobian_determinant": float(
                np.min(final.operator.jacobian_determinants(cube))
            ),
        },
        "development_readout_passed": passed,
        "development_champion": selected if passed else None,
        "validation_opened": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_triangular_conditional_capacity"]
