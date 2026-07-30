"""Grouped FilmMatch capacity audit for bounded local Gaussian residuals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.code_domain_gaussian_residual import (
    CodeDomainGaussianResidualOperator,
    fit_code_domain_gaussian_residual,
)


def _grid(size: int, margin: float) -> np.ndarray:
    axis = np.linspace(margin, 1.0 - margin, size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _jacobians(
    operator: CodeDomainGaussianResidualOperator,
    points: np.ndarray,
    *,
    strength: float,
    step: float,
) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (
                operator.apply(points + offset, strength=strength)
                - operator.apply(points - offset, strength=strength)
            )
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


def structural_strength(
    operator: CodeDomainGaussianResidualOperator,
    config: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Choose the first frozen strength satisfying training-only structure."""

    audit = config["candidate"]["structural_audit"]
    cube = _grid(int(audit["cube_size"]), 0.0)
    interior = _grid(
        int(audit["jacobian_size"]),
        float(audit["jacobian_margin"]),
    )
    candidates: list[dict[str, Any]] = []
    selected = 0.0
    for strength in map(float, audit["strength_schedule"]):
        mapped = operator.apply(cube, strength=strength)
        jacobian = _jacobians(
            operator,
            interior,
            strength=strength,
            step=float(audit["jacobian_step"]),
        )
        determinants = np.linalg.det(jacobian)
        norms = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
        row = {
            "strength": strength,
            "finite": bool(np.all(np.isfinite(mapped))),
            "minimum_output": float(np.min(mapped)),
            "maximum_output": float(np.max(mapped)),
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "maximum_jacobian_spectral_norm": float(np.max(norms)),
        }
        row["safe"] = bool(
            row["finite"]
            and row["minimum_output"] >= 0.0
            and row["maximum_output"] <= 1.0
            and row["minimum_jacobian_determinant"]
            >= float(audit["minimum_jacobian_determinant"])
            and row["maximum_jacobian_spectral_norm"]
            <= float(audit["maximum_jacobian_spectral_norm"])
        )
        candidates.append(row)
        if row["safe"] and selected == 0.0:
            selected = strength
    return selected, {"candidates": candidates, "selected_strength": selected}


def _fit_variant(
    source: np.ndarray,
    target: np.ndarray,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[CodeDomainGaussianResidualOperator, float, dict[str, Any]]:
    operator = fit_code_domain_gaussian_residual(
        source,
        target,
        axis_size=int(variant["axis_size"]),
        sigma=float(variant["sigma"]),
        epsilon=float(config["candidate"]["epsilon"]),
        ridge=float(variant["ridge"]),
    )
    strength, audit = structural_strength(operator, config)
    return operator, strength, audit


def _fold(
    *,
    fold_id: str,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    held_source: np.ndarray,
    held_target: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for variant in config["candidate"]["variants"]:
        operator, strength, audit = _fit_variant(
            fit_source, fit_target, variant, config
        )
        prediction = operator.apply(held_source, strength=strength)
        rows[str(variant["id"])] = {
            "metrics": prediction_metrics(prediction, held_target),
            "selected_structural_strength": strength,
            "structural_audit": audit,
        }
    return {
        "fold_id": fold_id,
        "development_samples": int(len(fit_source)),
        "held_samples": int(len(held_source)),
        "variants": rows,
    }


def _aggregate(folds: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in variants:
        rmses = np.asarray(
            [row["variants"][variant]["metrics"]["rgb_rmse"] for row in folds]
        )
        strengths = np.asarray(
            [
                row["variants"][variant]["selected_structural_strength"]
                for row in folds
            ]
        )
        output[variant] = {
            "median_rgb_rmse": float(np.median(rmses)),
            "mean_rgb_rmse": float(np.mean(rmses)),
            "worst_rgb_rmse": float(np.max(rmses)),
            "minimum_structural_strength": float(np.min(strengths)),
            "median_structural_strength": float(np.median(strengths)),
        }
    return output


def evaluate_gaussian_capacity(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reflective_source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    reflective_target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    emissive_source = np.asarray(datasets["emissive_source"], dtype=np.float64)
    emissive_target = np.asarray(datasets["emissive_target"], dtype=np.float64)
    illuminants = np.asarray(
        [row["illuminant"] for row in datasets["reflective_records"]]
    )
    hues = np.asarray([row["hue_sector"] for row in datasets["emissive_records"]])
    reflective_folds = [
        _fold(
            fold_id=f"held-illuminant-{illuminant}",
            fit_source=reflective_source[illuminants != illuminant],
            fit_target=reflective_target[illuminants != illuminant],
            held_source=reflective_source[illuminants == illuminant],
            held_target=reflective_target[illuminants == illuminant],
            config=config,
        )
        for illuminant in ("5600K", "3200K", "5600K_CTB")
    ]
    hue_folds = [
        _fold(
            fold_id=f"held-emissive-hue-{hue}",
            fit_source=np.concatenate(
                [reflective_source, emissive_source[hues != hue]]
            ),
            fit_target=np.concatenate(
                [reflective_target, emissive_target[hues != hue]]
            ),
            held_source=emissive_source[hues == hue],
            held_target=emissive_target[hues == hue],
            config=config,
        )
        for hue in sorted(set(hues))
    ]
    names = [str(row["id"]) for row in config["candidate"]["variants"]]
    reflective = _aggregate(reflective_folds, names)
    emissive = _aggregate(hue_folds, names)
    parent = config["parent_capacity_result"]
    identity_reflective = float(parent["reflective_identity_median_rmse"])
    affine_reflective = float(parent["reflective_affine_median_rmse"])
    identity_hue = float(parent["emissive_identity_median_rmse"])
    for name in names:
        reflective[name]["median_improvement_over_identity"] = float(
            1.0 - reflective[name]["median_rgb_rmse"] / identity_reflective
        )
        reflective[name]["median_improvement_over_affine"] = float(
            1.0 - reflective[name]["median_rgb_rmse"] / affine_reflective
        )
        emissive[name]["median_improvement_over_identity"] = float(
            1.0 - emissive[name]["median_rgb_rmse"] / identity_hue
        )
    scores = {
        name: reflective[name]["mean_rgb_rmse"] + emissive[name]["mean_rgb_rmse"]
        for name in names
    }
    selected = min(names, key=lambda name: (scores[name], names.index(name)))
    gates = config["development_readout"]
    passed = bool(
        reflective[selected]["median_improvement_over_identity"]
        >= float(gates["minimum_reflective_improvement_over_identity"])
        and reflective[selected]["median_improvement_over_affine"]
        >= float(gates["minimum_reflective_improvement_over_affine"])
        and emissive[selected]["median_improvement_over_identity"]
        >= float(gates["minimum_emissive_improvement_over_identity"])
        and reflective[selected]["minimum_structural_strength"] > 0.0
        and emissive[selected]["minimum_structural_strength"] > 0.0
    )
    combined_source = np.concatenate([reflective_source, emissive_source])
    combined_target = np.concatenate([reflective_target, emissive_target])
    variant = next(
        row for row in config["candidate"]["variants"] if row["id"] == selected
    )
    final_operator, final_strength, final_audit = _fit_variant(
        combined_source, combined_target, variant, config
    )
    report = {
        "schema": "neuro_film.u5_r2aw2_filmmatch_gaussian_capacity.v1",
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_folds,
        "emissive_hue_folds": hue_folds,
        "reflective_aggregate": reflective,
        "emissive_hue_aggregate": emissive,
        "selection": {
            "selected": selected,
            "combined_mean_rmse_score": scores,
        },
        "final_operator": final_operator.to_dict(),
        "final_structural_strength": final_strength,
        "final_structural_audit": final_audit,
        "development_readout_passed": passed,
        "development_champion": selected if passed else None,
        "validation_opened": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "evaluate_gaussian_capacity",
    "structural_strength",
]
