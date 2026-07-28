"""AO8D split-sensitivity and within-domain permutation confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.velvia_colourwise_two_expert_cv import (
    deterministic_domain_folds,
    fit_proxy_operator,
    validate_contract as validate_ao8b_contract,
)
from src.eval.velvia_crossfit_case_selector import (
    case_knn_labels,
    inner_crossfit_labels,
    validate_contract as validate_ao8c_contract,
)
from src.real_film.combined_velvia_operator import load_combined_velvia_pairs
from src.real_film.velvia_chart_explainability import _fit_record, _metrics


class VelviaCaseResidualConfirmationError(ValueError):
    """Raised when AO8D lineage, splits, or controls drift."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_json(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if _sha256(path) != spec["sha256"]:
        raise VelviaCaseResidualConfirmationError(
            f"lineage hash mismatch: {spec['path']}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate AO8D and return the exact AO8C contract."""

    gates = config["gates"]
    split_ids = [row["split_id"] for row in config["split_repetitions"]]
    fixed = config["fixed_candidate"]
    if (
        config.get("experiment_id")
        != "u5.r2ao8d-velvia-case-residual-domain-confirmation-v1"
        or split_ids != ["split_a", "split_b", "split_c"]
        or fixed
        != {
            "candidate_id": "case_knn_k3_distance",
            "neighbors": 3,
            "vote": "inverse_distance",
            "selection_allowed": False,
        }
        or int(config["evaluation"]["outer_fold_count"]) != 5
        or int(config["evaluation"]["inner_fold_count"]) != 4
        or int(
            config["evaluation"]["within_domain_shuffled_label_replicates"]
        )
        != 128
        or float(
            gates[
                "minimum_candidate_rgb_rmse_gain_over_domain_control_median"
            ]
        )
        != 0.03
        or int(gates["minimum_passing_splits_for_rgb_and_domain_gains"]) != 2
        or float(
            gates[
                "maximum_within_domain_shuffle_empirical_p_value_each_split"
            ]
        )
        != 0.05
        or not config["operator_fitting_allowed"]
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or config["novel_case_retrieval_claim_allowed"]
    ):
        raise VelviaCaseResidualConfirmationError(
            "AO8D frozen contract mismatch"
        )
    parent = _load_exact_json(root, config["parent_decision"])
    if (
        parent.get("decision")
        != config["parent_decision"]["required_decision"]
        or parent["selected_candidate"]["candidate_id"]
        != config["parent_decision"]["fixed_candidate_id"]
    ):
        raise VelviaCaseResidualConfirmationError(
            "AO8D parent decision mismatch"
        )
    ao8c = _load_exact_json(root, config["inputs"]["ao8c_contract"])
    validate_ao8c_contract(root, ao8c)
    candidates = {
        row["candidate_id"]: row for row in ao8c["candidate_bank"]
    }
    if candidates.get(fixed["candidate_id"]) != {
        "candidate_id": fixed["candidate_id"],
        "neighbors": fixed["neighbors"],
        "vote": fixed["vote"],
    }:
        raise VelviaCaseResidualConfirmationError(
            "AO8D fixed candidate mismatch"
        )
    return ao8c


def _compact_fit(result: Any) -> dict[str, Any]:
    record = _fit_record(result)
    return {
        "converged": record["converged"],
        "development_rgb_rmse": record["development_rgb_rmse"],
        "operator_sha256": record["operator_sha256"],
        "restart_index": record["restart_index"],
    }


def _gain(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise VelviaCaseResidualConfirmationError(
            "comparison denominator is invalid"
        )
    return 1.0 - value / baseline


def _evaluate_split(
    *,
    chart_source: np.ndarray,
    chart_target: np.ndarray,
    palette_source: np.ndarray,
    palette_target: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    ao8c: Mapping[str, Any],
    ao8b: Mapping[str, Any],
) -> dict[str, Any]:
    outer_count = int(config["evaluation"]["outer_fold_count"])
    chart_folds = deterministic_domain_folds(
        len(chart_source), outer_count, int(split["outer_chart_seed"])
    )
    palette_folds = deterministic_domain_folds(
        len(palette_source), outer_count, int(split["outer_palette_seed"])
    )
    combined_folds = np.concatenate((chart_folds, palette_folds))
    predictions = {
        name: np.empty_like(source, dtype=np.float64)
        for name in ("candidate", "combined", "oracle", "domain")
    }
    replicate_count = int(
        config["evaluation"]["within_domain_shuffled_label_replicates"]
    )
    shuffled = [
        np.empty_like(source, dtype=np.float64)
        for _ in range(replicate_count)
    ]
    selected_labels = np.empty(len(source), dtype=np.int8)
    fold_records = []
    all_converged = True
    candidate = config["fixed_candidate"]
    for outer_fold in range(outer_count):
        chart_indices = np.flatnonzero(chart_folds != outer_fold)
        palette_indices = np.flatnonzero(palette_folds != outer_fold)
        test = combined_folds == outer_fold
        development = ~test
        chart_fit = fit_proxy_operator(
            chart_source[chart_indices],
            chart_target[chart_indices],
            ao8b,
        )
        palette_fit = fit_proxy_operator(
            palette_source[palette_indices],
            palette_target[palette_indices],
            ao8b,
        )
        combined_fit = fit_proxy_operator(
            source[development], target[development], ao8b
        )
        inner_config = json.loads(json.dumps(ao8c))
        inner_config["inner_label_crossfit"]["chart_seed_base"] = int(
            split["inner_chart_seed_base"]
        )
        inner_config["inner_label_crossfit"]["palette_seed_base"] = int(
            split["inner_palette_seed_base"]
        )
        case_source, case_labels, inner_records, inner_converged = (
            inner_crossfit_labels(
                outer_fold=outer_fold,
                chart_source=chart_source,
                chart_target=chart_target,
                palette_source=palette_source,
                palette_target=palette_target,
                chart_outer_indices=chart_indices,
                palette_outer_indices=palette_indices,
                config=inner_config,
                fit_config=ao8b,
            )
        )
        query = source[test]
        chart_output = chart_fit.operator.apply(query)
        palette_output = palette_fit.operator.apply(query)
        combined_output = combined_fit.operator.apply(query)
        labels = case_knn_labels(
            query,
            case_source,
            case_labels,
            neighbors=int(candidate["neighbors"]),
            vote=str(candidate["vote"]),
        )
        selected_labels[test] = labels
        predictions["candidate"][test] = np.where(
            labels[:, None] == 1, chart_output, palette_output
        )
        predictions["combined"][test] = combined_output
        oracle_labels = (
            np.sum((chart_output - target[test]) ** 2, axis=1)
            <= np.sum((palette_output - target[test]) ** 2, axis=1)
        ).astype(np.int8)
        predictions["oracle"][test] = np.where(
            oracle_labels[:, None] == 1, chart_output, palette_output
        )
        domain_labels = np.concatenate(
            (
                np.ones(len(chart_indices), dtype=np.int8),
                np.zeros(len(palette_indices), dtype=np.int8),
            )
        )
        predicted_domain = case_knn_labels(
            query,
            case_source,
            domain_labels,
            neighbors=int(candidate["neighbors"]),
            vote=str(candidate["vote"]),
        )
        predictions["domain"][test] = np.where(
            predicted_domain[:, None] == 1,
            chart_output,
            palette_output,
        )
        chart_case_count = len(chart_indices)
        for replicate in range(replicate_count):
            rng = np.random.default_rng(
                int(split["within_domain_shuffle_seed"])
                + replicate * outer_count
                + outer_fold
            )
            permuted = case_labels.copy()
            permuted[:chart_case_count] = rng.permutation(
                permuted[:chart_case_count]
            )
            permuted[chart_case_count:] = rng.permutation(
                permuted[chart_case_count:]
            )
            shuffled_labels = case_knn_labels(
                query,
                case_source,
                permuted,
                neighbors=int(candidate["neighbors"]),
                vote=str(candidate["vote"]),
            )
            shuffled[replicate][test] = np.where(
                shuffled_labels[:, None] == 1,
                chart_output,
                palette_output,
            )
        outer_converged = bool(
            chart_fit.converged
            and palette_fit.converged
            and combined_fit.converged
        )
        all_converged = (
            all_converged and outer_converged and inner_converged
        )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "outer_chart_fit": _compact_fit(chart_fit),
                "outer_palette_fit": _compact_fit(palette_fit),
                "outer_combined_fit": _compact_fit(combined_fit),
                "outer_converged": outer_converged,
                "inner_converged": inner_converged,
                "inner_operator_sha256": [
                    {
                        "inner_fold": row["inner_fold"],
                        "chart": row["chart_fit"]["operator_sha256"],
                        "palette": row["palette_fit"]["operator_sha256"],
                    }
                    for row in inner_records
                ],
                "crossfit_chart_labels": int(np.sum(case_labels)),
                "crossfit_palette_labels": int(np.sum(case_labels == 0)),
                "crossfit_label_domain_agreement": float(
                    np.mean(case_labels == domain_labels)
                ),
                "held_out_chart_selections": int(np.sum(labels)),
                "held_out_palette_selections": int(np.sum(labels == 0)),
                "held_out_oracle_agreement": float(
                    np.mean(labels == oracle_labels)
                ),
            }
        )
    scores = {
        name: _metrics(values, target)
        for name, values in predictions.items()
    }
    shuffled_rmse = [
        _metrics(values, target)["rgb_rmse"] for values in shuffled
    ]
    pseudocount = int(config["evaluation"]["empirical_pseudocount"])
    shuffle_p = (
        pseudocount
        + sum(value <= scores["candidate"]["rgb_rmse"] for value in shuffled_rmse)
    ) / (replicate_count + pseudocount)
    candidate_score = scores["candidate"]
    combined_score = scores["combined"]
    return {
        "split_id": split["split_id"],
        "fold_assignments": {
            "chart": chart_folds.tolist(),
            "palette": palette_folds.tolist(),
        },
        "folds": fold_records,
        "scores": scores,
        "gains": {
            "oracle_rgb_rmse_over_combined": _gain(
                scores["oracle"]["rgb_rmse"],
                combined_score["rgb_rmse"],
            ),
            "candidate_rgb_rmse_over_combined": _gain(
                candidate_score["rgb_rmse"],
                combined_score["rgb_rmse"],
            ),
            "candidate_mean_delta_e76_over_combined": _gain(
                candidate_score["mean_delta_e76"],
                combined_score["mean_delta_e76"],
            ),
            "candidate_rgb_rmse_over_domain": _gain(
                candidate_score["rgb_rmse"],
                scores["domain"]["rgb_rmse"],
            ),
            "candidate_to_combined_maximum_absolute_error_ratio": (
                candidate_score["rgb_maximum_absolute_error"]
                / combined_score["rgb_maximum_absolute_error"]
            ),
        },
        "candidate_chart_selection_fraction": float(
            np.mean(selected_labels == 1)
        ),
        "candidate_palette_selection_fraction": float(
            np.mean(selected_labels == 0)
        ),
        "within_domain_shuffle": {
            "replicates": replicate_count,
            "minimum_rgb_rmse": float(np.min(shuffled_rmse)),
            "median_rgb_rmse": float(np.median(shuffled_rmse)),
            "maximum_rgb_rmse": float(np.max(shuffled_rmse)),
            "empirical_p_value": shuffle_p,
        },
        "all_fits_converged": all_converged,
    }


def evaluate_case_residual_domain_confirmation(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the fixed case selector across three new split systems."""

    ao8c = validate_contract(root, config)
    ao8b = validate_ao8c_contract(root, ao8c)
    ao5 = validate_ao8b_contract(root, ao8b)
    pairs = load_combined_velvia_pairs(
        root / ao5["inputs"]["chart_pairs"],
        root / ao5["inputs"]["palette_pairs"],
        ao5,
    )
    chart_source, chart_target = pairs["velvia_chart"]
    palette_source, palette_target = pairs["velvia_palette"]
    source, target = pairs["combined"]
    splits = [
        _evaluate_split(
            chart_source=chart_source,
            chart_target=chart_target,
            palette_source=palette_source,
            palette_target=palette_target,
            source=source,
            target=target,
            split=split,
            config=config,
            ao8c=ao8c,
            ao8b=ao8b,
        )
        for split in config["split_repetitions"]
    ]
    gates = config["gates"]
    oracle_gains = [
        row["gains"]["oracle_rgb_rmse_over_combined"] for row in splits
    ]
    rgb_gains = [
        row["gains"]["candidate_rgb_rmse_over_combined"] for row in splits
    ]
    perceptual_gains = [
        row["gains"]["candidate_mean_delta_e76_over_combined"]
        for row in splits
    ]
    domain_gains = [
        row["gains"]["candidate_rgb_rmse_over_domain"] for row in splits
    ]
    shuffle_p = [
        row["within_domain_shuffle"]["empirical_p_value"] for row in splits
    ]
    maximum_error_ratios = [
        row["gains"][
            "candidate_to_combined_maximum_absolute_error_ratio"
        ]
        for row in splits
    ]
    minimum_split_count = int(
        gates["minimum_passing_splits_for_rgb_and_domain_gains"]
    )
    checks = {
        "oracle_value_each_split": all(
            value
            >= float(
                gates[
                    "minimum_oracle_rgb_rmse_gain_over_combined_each_split"
                ]
            )
            for value in oracle_gains
        ),
        "candidate_rgb_gain_median": float(np.median(rgb_gains))
        >= float(
            gates["minimum_candidate_rgb_rmse_gain_over_combined_median"]
        ),
        "candidate_rgb_gain_split_count": sum(
            value
            >= float(
                gates[
                    "minimum_candidate_rgb_rmse_gain_over_combined_median"
                ]
            )
            for value in rgb_gains
        )
        >= minimum_split_count,
        "candidate_perceptual_gain_median": float(
            np.median(perceptual_gains)
        )
        >= float(
            gates[
                "minimum_candidate_mean_delta_e76_gain_over_combined_median"
            ]
        ),
        "candidate_domain_gain_median": float(np.median(domain_gains))
        >= float(
            gates[
                "minimum_candidate_rgb_rmse_gain_over_domain_control_median"
            ]
        ),
        "candidate_domain_gain_split_count": sum(
            value
            >= float(
                gates[
                    "minimum_candidate_rgb_rmse_gain_over_domain_control_median"
                ]
            )
            for value in domain_gains
        )
        >= minimum_split_count,
        "within_domain_shuffle_each_split": all(
            value
            <= float(
                gates[
                    "maximum_within_domain_shuffle_empirical_p_value_each_split"
                ]
            )
            for value in shuffle_p
        ),
        "maximum_error_each_split": all(
            value
            <= float(
                gates[
                    "maximum_candidate_to_combined_maximum_absolute_error_ratio_each_split"
                ]
            )
            for value in maximum_error_ratios
        ),
        "both_experts_each_split": all(
            min(
                row["candidate_chart_selection_fraction"],
                row["candidate_palette_selection_fraction"],
            )
            >= float(gates["minimum_candidate_fraction_per_expert_each_split"])
            for row in splits
        ),
        "candidate_in_cube_each_split": all(
            row["scores"]["candidate"]["raw_out_of_cube_fraction"]
            <= float(gates["maximum_raw_out_of_cube_fraction"])
            for row in splits
        ),
        "all_fits_converged": all(
            row["all_fits_converged"] for row in splits
        ),
    }
    automatic_pass = all(checks.values())
    if automatic_pass:
        decision = config["decision_if_pass"]
    elif not checks["within_domain_shuffle_each_split"]:
        decision = config["decision_if_within_domain_shuffle_fails"]
    else:
        decision = config["decision_if_split_stability_fails"]
    return {
        "schema": "neuro-film.u5.r2ao8d.velvia-case-residual-domain-confirmation-report.v1",
        "experiment_id": config["experiment_id"],
        "fixed_candidate_id": config["fixed_candidate"]["candidate_id"],
        "splits": splits,
        "aggregate": {
            "oracle_rgb_rmse_gain_median": float(np.median(oracle_gains)),
            "candidate_rgb_rmse_gain_median": float(np.median(rgb_gains)),
            "candidate_mean_delta_e76_gain_median": float(
                np.median(perceptual_gains)
            ),
            "candidate_rgb_rmse_gain_over_domain_median": float(
                np.median(domain_gains)
            ),
            "within_domain_shuffle_p_maximum": float(np.max(shuffle_p)),
            "maximum_error_ratio_maximum": float(
                np.max(maximum_error_ratios)
            ),
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "VelviaCaseResidualConfirmationError",
    "evaluate_case_residual_domain_confirmation",
    "validate_contract",
]
