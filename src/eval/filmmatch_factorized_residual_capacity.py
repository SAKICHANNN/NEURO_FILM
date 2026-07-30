"""Grouped capacity for a safe global base plus local bounded residual."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256
from src.roll2film.factorized_code_domain_residual import (
    FactorizedCodeDomainResidualOperator,
    fit_factorized_code_domain_residual,
)
from src.roll2film.monotone_curve_matrix import (
    fit_monotone_curve_positive_matrix,
)


class _StrengthOperator(Protocol):
    def apply(self, rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray: ...


def _grid(size: int, margin: float) -> np.ndarray:
    axis = np.linspace(margin, 1.0 - margin, size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _structural_strength(
    operator: _StrengthOperator,
    config: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    audit = config["candidate"]["structural_audit"]
    cube = _grid(int(audit["cube_size"]), 0.0)
    interior = _grid(
        int(audit["jacobian_size"]), float(audit["jacobian_margin"])
    )
    step = float(audit["jacobian_step"])
    rows = []
    selected = 0.0
    for strength in map(float, audit["strength_schedule"]):
        mapped = operator.apply(cube, strength=strength)
        columns = []
        for channel in range(3):
            offset = np.zeros(3)
            offset[channel] = step
            columns.append(
                (
                    operator.apply(interior + offset, strength=strength)
                    - operator.apply(interior - offset, strength=strength)
                )
                / (2.0 * step)
            )
        jacobian = np.stack(columns, axis=-1)
        determinants = np.linalg.det(jacobian)
        norms = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
        row = {
            "strength": strength,
            "minimum_output": float(np.min(mapped)),
            "maximum_output": float(np.max(mapped)),
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "maximum_jacobian_spectral_norm": float(np.max(norms)),
        }
        row["safe"] = bool(
            np.all(np.isfinite(mapped))
            and row["minimum_output"] >= 0.0
            and row["maximum_output"] <= 1.0
            and row["minimum_jacobian_determinant"]
            >= float(audit["minimum_jacobian_determinant"])
            and row["maximum_jacobian_spectral_norm"]
            <= float(audit["maximum_jacobian_spectral_norm"])
        )
        rows.append(row)
        if row["safe"] and selected == 0.0:
            selected = strength
    return selected, {"selected_strength": selected, "candidates": rows}


def _fit_base(
    source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
) -> Any:
    fit = config["candidate"]["base_fit"]
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


def _fold(
    *,
    fold_id: str,
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    held_source: np.ndarray,
    held_target: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base = _fit_base(fit_source, fit_target, config)
    rows = {
        "global_base": {
            "metrics": prediction_metrics(base.apply(held_source), held_target),
            "selected_structural_strength": 1.0,
        }
    }
    for variant in config["candidate"]["variants"]:
        operator = fit_factorized_code_domain_residual(
            base,
            fit_source,
            fit_target,
            axis_size=int(variant["axis_size"]),
            sigma=float(variant["sigma"]),
            epsilon=float(config["candidate"]["epsilon"]),
            ridge=float(variant["ridge"]),
        )
        strength, audit = _structural_strength(operator, config)
        rows[str(variant["id"])] = {
            "metrics": prediction_metrics(
                operator.apply(held_source, strength=strength), held_target
            ),
            "selected_structural_strength": strength,
            "structural_audit": audit,
        }
    return {"fold_id": fold_id, "variants": rows}


def _aggregate(folds: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    output = {}
    for name in names:
        rmse = np.asarray(
            [row["variants"][name]["metrics"]["rgb_rmse"] for row in folds]
        )
        strength = np.asarray(
            [
                row["variants"][name]["selected_structural_strength"]
                for row in folds
            ]
        )
        output[name] = {
            "median_rgb_rmse": float(np.median(rmse)),
            "mean_rgb_rmse": float(np.mean(rmse)),
            "worst_rgb_rmse": float(np.max(rmse)),
            "minimum_structural_strength": float(np.min(strength)),
            "median_structural_strength": float(np.median(strength)),
        }
    return output


def evaluate_factorized_capacity(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rs = np.asarray(datasets["reflective_source"], dtype=np.float64)
    rt = np.asarray(datasets["reflective_target"], dtype=np.float64)
    es = np.asarray(datasets["emissive_source"], dtype=np.float64)
    et = np.asarray(datasets["emissive_target"], dtype=np.float64)
    illuminants = np.asarray(
        [row["illuminant"] for row in datasets["reflective_records"]]
    )
    hues = np.asarray([row["hue_sector"] for row in datasets["emissive_records"]])
    reflective_folds = [
        _fold(
            fold_id=f"held-illuminant-{name}",
            fit_source=rs[illuminants != name],
            fit_target=rt[illuminants != name],
            held_source=rs[illuminants == name],
            held_target=rt[illuminants == name],
            config=config,
        )
        for name in ("5600K", "3200K", "5600K_CTB")
    ]
    hue_folds = [
        _fold(
            fold_id=f"held-emissive-hue-{hue}",
            fit_source=np.concatenate([rs, es[hues != hue]]),
            fit_target=np.concatenate([rt, et[hues != hue]]),
            held_source=es[hues == hue],
            held_target=et[hues == hue],
            config=config,
        )
        for hue in sorted(set(hues))
    ]
    names = ["global_base"] + [
        str(row["id"]) for row in config["candidate"]["variants"]
    ]
    reflective = _aggregate(reflective_folds, names)
    emissive = _aggregate(hue_folds, names)
    parent = config["parent_capacity_result"]
    for name in names:
        reflective[name]["median_improvement_over_identity"] = float(
            1.0
            - reflective[name]["median_rgb_rmse"]
            / float(parent["reflective_identity_median_rmse"])
        )
        reflective[name]["median_improvement_over_affine"] = float(
            1.0
            - reflective[name]["median_rgb_rmse"]
            / float(parent["reflective_affine_median_rmse"])
        )
        emissive[name]["median_improvement_over_identity"] = float(
            1.0
            - emissive[name]["median_rgb_rmse"]
            / float(parent["emissive_identity_median_rmse"])
        )
    candidate_names = names[1:]
    score = {
        name: reflective[name]["mean_rgb_rmse"] + emissive[name]["mean_rgb_rmse"]
        for name in candidate_names
    }
    selected = min(
        candidate_names,
        key=lambda name: (score[name], candidate_names.index(name)),
    )
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
    base = _fit_base(np.concatenate([rs, es]), np.concatenate([rt, et]), config)
    variant = next(
        row for row in config["candidate"]["variants"] if row["id"] == selected
    )
    final = fit_factorized_code_domain_residual(
        base,
        np.concatenate([rs, es]),
        np.concatenate([rt, et]),
        axis_size=int(variant["axis_size"]),
        sigma=float(variant["sigma"]),
        epsilon=float(config["candidate"]["epsilon"]),
        ridge=float(variant["ridge"]),
    )
    final_strength, final_audit = _structural_strength(final, config)
    report = {
        "schema": "neuro_film.u5_r2aw3_filmmatch_factorized_capacity.v1",
        "experiment_id": config["experiment_id"],
        "reflective_folds": reflective_folds,
        "emissive_hue_folds": hue_folds,
        "reflective_aggregate": reflective,
        "emissive_hue_aggregate": emissive,
        "selection": {"selected": selected, "score": score},
        "final_operator": final.to_dict(),
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


__all__ = ["evaluate_factorized_capacity"]
