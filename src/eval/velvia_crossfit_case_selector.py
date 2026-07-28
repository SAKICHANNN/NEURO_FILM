"""Nested cross-fitted case retrieval between two explicit proxy operators."""

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
from src.real_film.combined_velvia_operator import load_combined_velvia_pairs
from src.real_film.velvia_chart_explainability import _fit_record, _metrics


class VelviaCrossfitCaseError(ValueError):
    """Raised when AO8C lineage, cross-fitting, or case routing drifts."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_json(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if _sha256(path) != spec["sha256"]:
        raise VelviaCrossfitCaseError(f"lineage hash mismatch: {spec['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate AO8C and return the already-validated AO8B contract."""

    candidate_ids = [row["candidate_id"] for row in config["candidate_bank"]]
    gates = config["gates"]
    negative = config["negative_controls"]
    if (
        config.get("experiment_id")
        != "u5.r2ao8c-velvia-crossfit-case-selector-v1"
        or candidate_ids
        != [
            "case_knn_k1_hard",
            "case_knn_k3_hard",
            "case_knn_k5_hard",
            "case_knn_k3_distance",
            "case_knn_k5_distance",
        ]
        or int(config["outer_evaluation"]["fold_count"]) != 5
        or int(config["inner_label_crossfit"]["fold_count"]) != 4
        or int(negative["shuffled_label_replicates"]) != 32
        or float(gates["minimum_oracle_rgb_rmse_gain_over_combined"]) != 0.1
        or float(gates["minimum_selected_rgb_rmse_gain_over_combined"]) != 0.05
        or float(
            gates["minimum_selected_mean_delta_e76_gain_over_combined"]
        )
        != 0.03
        or float(
            gates[
                "minimum_selected_rgb_rmse_gain_over_best_domain_only_control"
            ]
        )
        != 0.03
        or float(gates["maximum_shuffle_empirical_p_value"]) != 0.05
        or not config["operator_fitting_allowed"]
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or config["novel_case_retrieval_claim_allowed"]
    ):
        raise VelviaCrossfitCaseError("AO8C frozen contract mismatch")
    parent = _load_exact_json(root, config["parent_decision"])
    if (
        parent.get("decision")
        != config["parent_decision"]["required_decision"]
        or float(parent["oracle_diagnostic"]["rgb_rmse_gain_over_global"])
        != float(config["parent_decision"]["required_oracle_gain"])
    ):
        raise VelviaCrossfitCaseError("AO8C parent evidence mismatch")
    ao8b = _load_exact_json(root, config["inputs"]["ao8b_contract"])
    validate_ao8b_contract(root, ao8b)
    outer = config["outer_evaluation"]
    if (
        int(ao8b["nested_evaluation"]["fold_count"])
        != int(outer["fold_count"])
        or int(ao8b["nested_evaluation"]["chart_split_seed"])
        != int(outer["chart_split_seed"])
        or int(ao8b["nested_evaluation"]["palette_split_seed"])
        != int(outer["palette_split_seed"])
    ):
        raise VelviaCrossfitCaseError("AO8C outer split drift")
    return ao8b


def case_knn_labels(
    query: np.ndarray,
    cases: np.ndarray,
    labels: np.ndarray,
    *,
    neighbors: int,
    vote: str,
) -> np.ndarray:
    """Predict hard chart/palette labels from mature kNN case retrieval."""

    query = np.asarray(query, dtype=np.float64)
    cases = np.asarray(cases, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        query.ndim != 2
        or cases.ndim != 2
        or query.shape[1] != 3
        or cases.shape[1] != 3
        or labels.shape != (len(cases),)
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(cases))
        or not np.all(np.isin(labels, [0, 1]))
        or not 1 <= neighbors <= len(cases)
        or neighbors % 2 != 1
    ):
        raise VelviaCrossfitCaseError("kNN case inputs are invalid")
    squared = np.sum((query[:, None, :] - cases[None, :, :]) ** 2, axis=2)
    nearest = np.argsort(squared, axis=1, kind="stable")[:, :neighbors]
    selected_labels = labels[nearest]
    if vote == "uniform":
        scores = np.mean(selected_labels, axis=1)
    elif vote == "inverse_distance":
        distance = np.sqrt(np.take_along_axis(squared, nearest, axis=1))
        weights = 1.0 / np.maximum(distance, 1e-12)
        scores = np.sum(weights * selected_labels, axis=1) / np.sum(
            weights, axis=1
        )
    else:
        raise VelviaCrossfitCaseError("unsupported kNN vote")
    return (scores >= 0.5).astype(np.int8)


def _inner_crossfit_labels(
    *,
    outer_fold: int,
    chart_source: np.ndarray,
    chart_target: np.ndarray,
    palette_source: np.ndarray,
    palette_target: np.ndarray,
    chart_outer_indices: np.ndarray,
    palette_outer_indices: np.ndarray,
    config: Mapping[str, Any],
    fit_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], bool]:
    """Create target-derived labels only from inner held-out predictions."""

    inner = config["inner_label_crossfit"]
    fold_count = int(inner["fold_count"])
    chart_inner = deterministic_domain_folds(
        len(chart_outer_indices),
        fold_count,
        int(inner["chart_seed_base"]) + outer_fold,
    )
    palette_inner = deterministic_domain_folds(
        len(palette_outer_indices),
        fold_count,
        int(inner["palette_seed_base"]) + outer_fold,
    )
    case_source = np.concatenate(
        (
            chart_source[chart_outer_indices],
            palette_source[palette_outer_indices],
        )
    )
    labels = np.empty(len(case_source), dtype=np.int8)
    records: list[dict[str, Any]] = []
    all_converged = True
    palette_offset = len(chart_outer_indices)
    for inner_fold in range(fold_count):
        chart_dev_local = chart_inner != inner_fold
        palette_dev_local = palette_inner != inner_fold
        chart_fit = fit_proxy_operator(
            chart_source[chart_outer_indices[chart_dev_local]],
            chart_target[chart_outer_indices[chart_dev_local]],
            fit_config,
        )
        palette_fit = fit_proxy_operator(
            palette_source[palette_outer_indices[palette_dev_local]],
            palette_target[palette_outer_indices[palette_dev_local]],
            fit_config,
        )
        chart_test_local = np.flatnonzero(chart_inner == inner_fold)
        palette_test_local = np.flatnonzero(palette_inner == inner_fold)
        local_test = np.concatenate(
            (chart_test_local, palette_offset + palette_test_local)
        )
        target = np.concatenate(
            (
                chart_target[chart_outer_indices[chart_test_local]],
                palette_target[palette_outer_indices[palette_test_local]],
            )
        )
        query = case_source[local_test]
        chart_output = chart_fit.operator.apply(query)
        palette_output = palette_fit.operator.apply(query)
        labels[local_test] = (
            np.sum((chart_output - target) ** 2, axis=1)
            <= np.sum((palette_output - target) ** 2, axis=1)
        ).astype(np.int8)
        converged = bool(chart_fit.converged and palette_fit.converged)
        all_converged = all_converged and converged
        records.append(
            {
                "inner_fold": inner_fold,
                "held_out_chart_rows": len(chart_test_local),
                "held_out_palette_rows": len(palette_test_local),
                "chart_fit": _fit_record(chart_fit),
                "palette_fit": _fit_record(palette_fit),
                "converged": converged,
            }
        )
    if not np.all(np.isin(labels, [0, 1])):
        raise VelviaCrossfitCaseError("inner cross-fit label coverage failed")
    return case_source, labels, records, all_converged


def evaluate_crossfit_case_selector(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Run exact outer CV, inner label cross-fit, and nuisance controls."""

    ao8b = validate_contract(root, config)
    ao5 = validate_ao8b_contract(root, ao8b)
    pairs = load_combined_velvia_pairs(
        root / ao5["inputs"]["chart_pairs"],
        root / ao5["inputs"]["palette_pairs"],
        ao5,
    )
    chart_source, chart_target = pairs["velvia_chart"]
    palette_source, palette_target = pairs["velvia_palette"]
    source, target = pairs["combined"]
    outer = config["outer_evaluation"]
    fold_count = int(outer["fold_count"])
    chart_folds = deterministic_domain_folds(
        len(chart_source), fold_count, int(outer["chart_split_seed"])
    )
    palette_folds = deterministic_domain_folds(
        len(palette_source), fold_count, int(outer["palette_split_seed"])
    )
    combined_folds = np.concatenate((chart_folds, palette_folds))
    specs = config["candidate_bank"]
    ids = [row["candidate_id"] for row in specs]
    base_ids = [
        "identity",
        "fixed_chart_expert",
        "fixed_palette_expert",
        "combined_global_operator",
        "two_expert_oracle",
    ]
    predictions = {
        name: np.empty_like(source, dtype=np.float64)
        for name in [*base_ids, *ids]
    }
    predicted_labels = {
        name: np.empty(len(source), dtype=np.int8) for name in ids
    }
    oracle_labels = np.empty(len(source), dtype=np.int8)
    domain_predictions = {
        name: np.empty_like(source, dtype=np.float64) for name in ids
    }
    replicate_count = int(
        config["negative_controls"]["shuffled_label_replicates"]
    )
    shuffled_predictions = [
        {
            name: np.empty_like(source, dtype=np.float64) for name in ids
        }
        for _ in range(replicate_count)
    ]
    selected_counts = {
        name: {"chart": 0, "palette": 0} for name in ids
    }
    fold_records: list[dict[str, Any]] = []
    all_fits_converged = True
    shuffle_seed = int(
        config["negative_controls"]["shuffled_label_seed"]
    )
    for outer_fold in range(fold_count):
        chart_outer_indices = np.flatnonzero(chart_folds != outer_fold)
        palette_outer_indices = np.flatnonzero(
            palette_folds != outer_fold
        )
        test = combined_folds == outer_fold
        development = ~test
        chart_fit = fit_proxy_operator(
            chart_source[chart_outer_indices],
            chart_target[chart_outer_indices],
            ao8b,
        )
        palette_fit = fit_proxy_operator(
            palette_source[palette_outer_indices],
            palette_target[palette_outer_indices],
            ao8b,
        )
        combined_fit = fit_proxy_operator(
            source[development], target[development], ao8b
        )
        case_source, case_labels, inner_records, inner_converged = (
            _inner_crossfit_labels(
                outer_fold=outer_fold,
                chart_source=chart_source,
                chart_target=chart_target,
                palette_source=palette_source,
                palette_target=palette_target,
                chart_outer_indices=chart_outer_indices,
                palette_outer_indices=palette_outer_indices,
                config=config,
                fit_config=ao8b,
            )
        )
        domain_labels = np.concatenate(
            (
                np.ones(len(chart_outer_indices), dtype=np.int8),
                np.zeros(len(palette_outer_indices), dtype=np.int8),
            )
        )
        query = source[test]
        chart_output = chart_fit.operator.apply(query)
        palette_output = palette_fit.operator.apply(query)
        combined_output = combined_fit.operator.apply(query)
        predictions["identity"][test] = query
        predictions["fixed_chart_expert"][test] = chart_output
        predictions["fixed_palette_expert"][test] = palette_output
        predictions["combined_global_operator"][test] = combined_output
        oracle = (
            np.sum((chart_output - target[test]) ** 2, axis=1)
            <= np.sum((palette_output - target[test]) ** 2, axis=1)
        ).astype(np.int8)
        oracle_labels[test] = oracle
        predictions["two_expert_oracle"][test] = np.where(
            oracle[:, None] == 1, chart_output, palette_output
        )
        fold_router_records = {}
        for spec in specs:
            candidate_id = spec["candidate_id"]
            labels = case_knn_labels(
                query,
                case_source,
                case_labels,
                neighbors=int(spec["neighbors"]),
                vote=str(spec["vote"]),
            )
            predicted_labels[candidate_id][test] = labels
            predictions[candidate_id][test] = np.where(
                labels[:, None] == 1, chart_output, palette_output
            )
            selected_counts[candidate_id]["chart"] += int(np.sum(labels))
            selected_counts[candidate_id]["palette"] += int(
                np.sum(labels == 0)
            )
            domain_labels_pred = case_knn_labels(
                query,
                case_source,
                domain_labels,
                neighbors=int(spec["neighbors"]),
                vote=str(spec["vote"]),
            )
            domain_predictions[candidate_id][test] = np.where(
                domain_labels_pred[:, None] == 1,
                chart_output,
                palette_output,
            )
            for replicate in range(replicate_count):
                rng = np.random.default_rng(
                    shuffle_seed + replicate * fold_count + outer_fold
                )
                permuted = rng.permutation(case_labels)
                shuffled = case_knn_labels(
                    query,
                    case_source,
                    permuted,
                    neighbors=int(spec["neighbors"]),
                    vote=str(spec["vote"]),
                )
                shuffled_predictions[replicate][candidate_id][test] = np.where(
                    shuffled[:, None] == 1,
                    chart_output,
                    palette_output,
                )
            fold_router_records[candidate_id] = {
                "held_out_chart_selections": int(np.sum(labels)),
                "held_out_palette_selections": int(np.sum(labels == 0)),
                "held_out_oracle_agreement": float(np.mean(labels == oracle)),
            }
        outer_converged = bool(
            chart_fit.converged
            and palette_fit.converged
            and combined_fit.converged
        )
        all_fits_converged = (
            all_fits_converged and outer_converged and inner_converged
        )
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "outer_chart_fit": _fit_record(chart_fit),
                "outer_palette_fit": _fit_record(palette_fit),
                "outer_combined_fit": _fit_record(combined_fit),
                "outer_converged": outer_converged,
                "inner_fits": inner_records,
                "inner_converged": inner_converged,
                "crossfit_chart_labels": int(np.sum(case_labels)),
                "crossfit_palette_labels": int(np.sum(case_labels == 0)),
                "crossfit_label_domain_agreement": float(
                    np.mean(case_labels == domain_labels)
                ),
                "routers": fold_router_records,
            }
        )

    score = {
        name: _metrics(values, target)
        for name, values in predictions.items()
    }
    for name in ids:
        score[name]["chart_selection_fraction"] = (
            selected_counts[name]["chart"] / len(source)
        )
        score[name]["palette_selection_fraction"] = (
            selected_counts[name]["palette"] / len(source)
        )
        score[name]["oracle_label_accuracy"] = float(
            np.mean(predicted_labels[name] == oracle_labels)
        )
    domain_scores = {
        name: _metrics(values, target)
        for name, values in domain_predictions.items()
    }
    shuffle_best: list[dict[str, Any]] = []
    for replicate, values_by_id in enumerate(shuffled_predictions):
        scores = {
            name: _metrics(values, target)
            for name, values in values_by_id.items()
        }
        best = min(ids, key=lambda name: (scores[name]["rgb_rmse"], name))
        shuffle_best.append(
            {
                "replicate": replicate,
                "candidate_id": best,
                "rgb_rmse": scores[best]["rgb_rmse"],
            }
        )
    selected_id = min(ids, key=lambda name: (score[name]["rgb_rmse"], name))
    domain_id = min(
        ids, key=lambda name: (domain_scores[name]["rgb_rmse"], name)
    )
    selected = score[selected_id]
    combined = score["combined_global_operator"]
    oracle = score["two_expert_oracle"]
    domain = domain_scores[domain_id]

    def gain(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            raise VelviaCrossfitCaseError("invalid gain denominator")
        return 1.0 - numerator / denominator

    gains = {
        "oracle_rgb_rmse_over_combined": gain(
            oracle["rgb_rmse"], combined["rgb_rmse"]
        ),
        "selected_rgb_rmse_over_combined": gain(
            selected["rgb_rmse"], combined["rgb_rmse"]
        ),
        "selected_mean_delta_e76_over_combined": gain(
            selected["mean_delta_e76"], combined["mean_delta_e76"]
        ),
        "selected_rgb_rmse_over_best_domain_only": gain(
            selected["rgb_rmse"], domain["rgb_rmse"]
        ),
        "selected_to_combined_maximum_absolute_error_ratio": (
            selected["rgb_maximum_absolute_error"]
            / combined["rgb_maximum_absolute_error"]
        ),
    }
    pseudocount = int(
        config["negative_controls"]["empirical_pseudocount"]
    )
    shuffle_p = (
        pseudocount
        + sum(
            row["rgb_rmse"] <= selected["rgb_rmse"]
            for row in shuffle_best
        )
    ) / (replicate_count + pseudocount)
    gates = config["gates"]
    domain_pass = gains["selected_rgb_rmse_over_best_domain_only"] >= float(
        gates[
            "minimum_selected_rgb_rmse_gain_over_best_domain_only_control"
        ]
    )
    shuffle_pass = shuffle_p <= float(
        gates["maximum_shuffle_empirical_p_value"]
    )
    checks = {
        "oracle_product_value": gains["oracle_rgb_rmse_over_combined"]
        >= float(gates["minimum_oracle_rgb_rmse_gain_over_combined"]),
        "selected_gain_over_combined": gains[
            "selected_rgb_rmse_over_combined"
        ]
        >= float(gates["minimum_selected_rgb_rmse_gain_over_combined"]),
        "selected_perceptual_gain_over_combined": gains[
            "selected_mean_delta_e76_over_combined"
        ]
        >= float(
            gates["minimum_selected_mean_delta_e76_gain_over_combined"]
        ),
        "beats_domain_only_control": domain_pass,
        "beats_shuffled_label_control": shuffle_pass,
        "selected_maximum_error_guard": gains[
            "selected_to_combined_maximum_absolute_error_ratio"
        ]
        <= float(
            gates[
                "maximum_selected_to_combined_maximum_absolute_error_ratio"
            ]
        ),
        "both_experts_selected": min(
            selected["chart_selection_fraction"],
            selected["palette_selection_fraction"],
        )
        >= float(gates["minimum_selected_fraction_per_expert"]),
        "selected_in_cube": selected["raw_out_of_cube_fraction"]
        <= float(gates["maximum_raw_out_of_cube_fraction"]),
        "all_outer_and_inner_fits_converged": all_fits_converged,
    }
    automatic_pass = all(checks.values())
    if automatic_pass:
        decision = config["decision_if_pass"]
    elif not domain_pass:
        decision = config["decision_if_domain_control_fails"]
    elif not shuffle_pass:
        decision = config["decision_if_shuffle_fails"]
    else:
        decision = config["decision_if_other_gate_fails"]
    return {
        "schema": "neuro-film.u5.r2ao8c.velvia-crossfit-case-selector-report.v1",
        "experiment_id": config["experiment_id"],
        "folds": fold_records,
        "scores": score,
        "selected_candidate_id": selected_id,
        "domain_only_control": {
            "selected_candidate_id": domain_id,
            "scores": domain_scores,
        },
        "shuffled_label_control": {
            "replicates": shuffle_best,
            "empirical_p_value": shuffle_p,
        },
        "gains": gains,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "VelviaCrossfitCaseError",
    "case_knn_labels",
    "evaluate_crossfit_case_selector",
    "validate_contract",
]
