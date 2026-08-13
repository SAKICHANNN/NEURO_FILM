"""CHAM7 bounded explicit operator on the registered Portra chart."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.portra400_chart_operator_d1 import _canonical, _load_samples, _sha
from src.roll2film.fixed_gaussian_residual import (
    FixedNeutralTrilinearLogOddsOperator,
    fit_fixed_neutral_trilinear_log_odds,
)
from src.roll2film.positive_film_fitting import fit_positive_film_response_operator

SCHEMA = "neuro-film.u5-r2cham7-portra400-bounded-operator-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham7-portra400-bounded-operator-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != SCHEMA
        or value.get("sampling")
        != "reuse_exact_cham6_linear_srgb_samples_blocks_and_folds"
        or value["models"]["candidate"].get("fitted_scalar_count") != 24
        or value["disclosure"].get("independent_confirmation_claim_allowed")
        is not False
    ):
        raise ValueError("unsupported CHAM7 contract")
    return value


def _rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(candidate - target))))


def _interior_cube(size: int, step: float) -> np.ndarray:
    axis = np.linspace(step, 1.0 - step, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _jacobians(
    operator: FixedNeutralTrilinearLogOddsOperator,
    points: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (operator.apply(points + offset) - operator.apply(points - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


def _fit(
    source: np.ndarray,
    target: np.ndarray,
    contract: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, FixedNeutralTrilinearLogOddsOperator]:
    base_config = contract["models"]["base"]
    candidate_config = contract["models"]["candidate"]
    base = fit_positive_film_response_operator(
        source,
        target,
        model="one_matrix",
        identity_mixture=float(base_config["identity_mixture"]),
        restart_count=int(base_config["restart_count"]),
        maximum_function_evaluations=int(base_config["maximum_function_evaluations"]),
        seed=seed,
    )
    candidate = fit_fixed_neutral_trilinear_log_odds(
        base.operator,
        source,
        target,
        ridge=float(candidate_config["ridge"]),
        fit_epsilon=float(candidate_config["fit_epsilon"]),
    )
    return base, candidate


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("CHAM7 parent drift")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            "required_decision" in binding
            and payload.get("decision") != binding["required_decision"]
        ):
            raise ValueError("CHAM7 parent decision drift")
        if (
            "required_stable_evidence_id" in binding
            and payload.get("stable_evidence_id")
            != binding["required_stable_evidence_id"]
        ):
            raise ValueError("CHAM7 parent stable identity drift")

    cham6_contract = json.loads(
        (root / contract["parents"]["cham6_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source, target, blocks, dataset = _load_samples(cham6_contract, root)
    fold_count = int(cham6_contract["sampling"]["fold_count"])
    audit = contract["audit"]
    step = float(audit["jacobian_step"])
    cube = _interior_cube(int(audit["jacobian_cube_size"]), step)
    neutral = np.repeat(
        np.linspace(0.0, 1.0, int(audit["neutral_axis_rows"]))[:, None], 3, axis=1
    )
    base_seed = int(contract["models"]["base"]["seed"])
    wrong_roll = int(contract["models"]["wrong_target_roll"])

    targets: list[np.ndarray] = []
    base_predictions: list[np.ndarray] = []
    candidate_predictions: list[np.ndarray] = []
    wrong_predictions: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []
    determinants: list[float] = []
    spectral_norms: list[float] = []
    neutral_errors: list[float] = []
    replay_errors: list[float] = []
    for fold in range(fold_count):
        held = blocks % fold_count == fold
        development = ~held
        base, candidate = _fit(
            source[development], target[development], contract, seed=base_seed + fold
        )
        wrong_target = np.roll(target[development], wrong_roll, axis=0)
        _, wrong = _fit(
            source[development], wrong_target, contract, seed=base_seed + 100 + fold
        )
        held_source = source[held]
        held_target = target[held]
        base_prediction = base.operator.apply(held_source)
        candidate_prediction = candidate.apply(held_source)
        replay = FixedNeutralTrilinearLogOddsOperator.from_dict(
            json.loads(json.dumps(candidate.to_dict(), sort_keys=True))
        ).apply(held_source)
        wrong_prediction = wrong.apply(held_source)
        jacobian = _jacobians(candidate, cube, step)
        fold_determinants = np.linalg.det(jacobian)
        fold_norms = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
        neutral_error = float(
            np.max(np.abs(candidate.apply(neutral) - base.operator.apply(neutral)))
        )
        replay_error = float(np.max(np.abs(replay - candidate_prediction)))
        targets.append(held_target)
        base_predictions.append(base_prediction)
        candidate_predictions.append(candidate_prediction)
        wrong_predictions.append(wrong_prediction)
        determinants.append(float(np.min(fold_determinants)))
        spectral_norms.append(float(np.max(fold_norms)))
        neutral_errors.append(neutral_error)
        replay_errors.append(replay_error)
        base_error = _rmse(base_prediction, held_target)
        candidate_error = _rmse(candidate_prediction, held_target)
        folds.append(
            {
                "fold": fold,
                "held_rows": int(np.count_nonzero(held)),
                "base_rmse": base_error,
                "candidate_rmse": candidate_error,
                "candidate_improvement_over_base_fraction": 1.0
                - candidate_error / base_error,
                "minimum_jacobian_determinant": determinants[-1],
                "maximum_jacobian_spectral_norm": spectral_norms[-1],
                "neutral_axis_residual": neutral_error,
                "replay_error": replay_error,
            }
        )

    combined_target = np.concatenate(targets)
    combined_base = np.concatenate(base_predictions)
    combined_candidate = np.concatenate(candidate_predictions)
    combined_wrong = np.concatenate(wrong_predictions)
    base_rmse = _rmse(combined_base, combined_target)
    candidate_rmse = _rmse(combined_candidate, combined_target)
    wrong_rmse = _rmse(combined_wrong, combined_target)
    fold_improvements = [
        float(row["candidate_improvement_over_base_fraction"]) for row in folds
    ]
    metrics = {
        "row_count": len(combined_target),
        "base_rmse": base_rmse,
        "candidate_rmse": candidate_rmse,
        "correspondence_shuffled_candidate_rmse": wrong_rmse,
        "improvement_over_base_fraction": 1.0 - candidate_rmse / base_rmse,
        "fold_win_fraction_over_base": float(
            np.mean(np.asarray(fold_improvements) > 0.0)
        ),
        "worst_fold_improvement_over_base_fraction": min(fold_improvements),
        "improvement_over_correspondence_shuffled_fraction": 1.0
        - candidate_rmse / wrong_rmse,
        "minimum_jacobian_determinant": min(determinants),
        "maximum_jacobian_spectral_norm": max(spectral_norms),
        "maximum_neutral_axis_residual": max(neutral_errors),
        "maximum_repeat_error": max(replay_errors),
    }
    gates = contract["gates"]
    checks = {
        "mean_improvement": metrics["improvement_over_base_fraction"]
        >= gates["minimum_improvement_over_base_fraction"],
        "fold_wins": metrics["fold_win_fraction_over_base"]
        >= gates["minimum_fold_win_fraction_over_base"],
        "tail": metrics["worst_fold_improvement_over_base_fraction"]
        >= gates["minimum_worst_fold_improvement_over_base_fraction"],
        "correspondence": metrics["improvement_over_correspondence_shuffled_fraction"]
        >= gates["minimum_improvement_over_correspondence_shuffled_fraction"],
        "accuracy": metrics["candidate_rmse"] <= gates["maximum_candidate_rmse"],
        "jacobian": metrics["minimum_jacobian_determinant"]
        > gates["minimum_jacobian_determinant_exclusive"],
        "jacobian_norm": metrics["maximum_jacobian_spectral_norm"]
        <= gates["maximum_jacobian_spectral_norm"],
        "neutral": metrics["maximum_neutral_axis_residual"]
        <= gates["maximum_neutral_axis_residual"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(value)) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "disclosure": contract["disclosure"],
        "dataset": dataset,
        "folds": folds,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
