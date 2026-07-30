"""Domain-balanced FilmMatch capacity for the fixed factorized operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.factorized_monotone_bernstein import (
    fit_factorized_monotone_bernstein,
)


def _domain_weights(
    reflective_count: int, emissive_count: int, emissive_share: float
) -> np.ndarray:
    if (
        reflective_count < 1
        or emissive_count < 1
        or not 0.0 < emissive_share < 1.0
    ):
        raise ValueError("invalid domain-balanced weight contract")
    weights = np.concatenate(
        (
            np.full(
                reflective_count,
                (1.0 - emissive_share) / reflective_count,
            ),
            np.full(emissive_count, emissive_share / emissive_count),
        )
    )
    return weights / np.mean(weights)


def _fit(
    source: np.ndarray,
    target: np.ndarray,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sample_weights: np.ndarray | None,
) -> Any:
    fit = config["candidate"]["fit"]
    return fit_factorized_monotone_bernstein(
        source,
        target,
        segment_count=int(fit["segment_count"]),
        curve_learned_mixture=float(fit["curve_learned_mixture"]),
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
        residual_degree=int(fit["residual_degree"]),
        residual_identity_ridge=float(fit["residual_identity_ridge"]),
        jacobian_floor=float(fit["jacobian_floor"]),
        safety_grid_size=int(fit["safety_grid_size"]),
        strength_steps=int(fit["strength_steps"]),
        maximum_residual_iterations=int(fit["maximum_residual_iterations"]),
        sample_weights=sample_weights,
    )


def _fold(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    held_source: np.ndarray,
    held_target: np.ndarray,
    *,
    fold_id: str,
    config: Mapping[str, Any],
    domain_counts: tuple[int, int] | None,
) -> dict[str, Any]:
    rows = {}
    for variant in config["candidate"]["variants"]:
        weights = None
        if domain_counts is not None:
            weights = _domain_weights(
                domain_counts[0],
                domain_counts[1],
                float(variant["emissive_loss_share"]),
            )
        fit = _fit(
            fit_source,
            fit_target,
            variant,
            config,
            sample_weights=weights,
        )
        rows[str(variant["id"])] = {
            "metrics": prediction_metrics(
                fit.operator.apply(held_source), held_target
            ),
            "converged": fit.converged,
            "development_rgb_rmse": fit.development_rgb_rmse,
            "safe_residual_strength": fit.operator.residual.strength,
        }
    return {"fold_id": fold_id, "variants": rows}


def _aggregate(folds: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        rmse = np.asarray(
            [row["variants"][name]["metrics"]["rgb_rmse"] for row in folds]
        )
        strengths = np.asarray(
            [row["variants"][name]["safe_residual_strength"] for row in folds]
        )
        result[name] = {
            "median_rgb_rmse": float(np.median(rmse)),
            "mean_rgb_rmse": float(np.mean(rmse)),
            "worst_rgb_rmse": float(np.max(rmse)),
            "median_safe_residual_strength": float(np.median(strengths)),
            "minimum_safe_residual_strength": float(np.min(strengths)),
            "all_converged": bool(
                all(row["variants"][name]["converged"] for row in folds)
            ),
        }
    return result


def evaluate_domain_balanced_capacity(
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
            domain_counts=None,
        )
        for name in ("5600K", "3200K", "5600K_CTB")
    ]
    hue_folds = []
    for hue in sorted(set(hues)):
        emissive_mask = hues != hue
        hue_folds.append(
            _fold(
                np.concatenate([rs, es[emissive_mask]]),
                np.concatenate([rt, et[emissive_mask]]),
                es[~emissive_mask],
                et[~emissive_mask],
                fold_id=f"held-emissive-hue-{hue}",
                config=config,
                domain_counts=(len(rs), int(np.sum(emissive_mask))),
            )
        )
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
    final_weights = _domain_weights(
        len(rs), len(es), float(variant["emissive_loss_share"])
    )
    final = _fit(
        np.concatenate([rs, es]),
        np.concatenate([rt, et]),
        variant,
        config,
        sample_weights=final_weights,
    )
    axis = np.linspace(0.01, 0.99, 9)
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    report = {
        "schema": "neuro_film.u5_r2ax5_filmmatch_domain_balanced.v1",
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_folds,
        "emissive_hue_folds": hue_folds,
        "reflective_aggregate": reflective,
        "emissive_hue_aggregate": emissive,
        "selection": {"selected": selected, "score": score},
        "final_fit": {
            "operator": final.operator.to_dict(),
            "development_rgb_rmse": final.development_rgb_rmse,
            "minimum_sampled_composite_jacobian_determinant": float(
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


__all__ = ["evaluate_domain_balanced_capacity"]
