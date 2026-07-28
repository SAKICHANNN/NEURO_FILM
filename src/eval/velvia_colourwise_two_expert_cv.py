"""Nested held-out audit for a bounded per-colour two-expert Velvia proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.real_film.combined_velvia_operator import load_combined_velvia_pairs
from src.real_film.velvia_chart_explainability import _fit_record, _metrics
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


class VelviaTwoExpertCVError(ValueError):
    """Raised when AO8B lineage, folds, or bounded routing drift."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact_json(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if _sha256(path) != spec["sha256"]:
        raise VelviaTwoExpertCVError(f"lineage hash mismatch: {spec['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen AO8B design and return the exact AO5 pair contract."""

    gates = config["gates"]
    router = config["router"]
    candidate_ids = [
        row["candidate_id"] for row in router["candidate_bank"]
    ]
    if (
        config.get("experiment_id")
        != "u5.r2ao8b-velvia-colourwise-two-expert-cv-v1"
        or int(config["nested_evaluation"]["fold_count"]) != 5
        or candidate_ids
        != [
            "nearest_support_hard",
            "nearest_support_soft_t025",
            "nearest_support_soft_t050",
            "nearest_support_soft_t100",
        ]
        or float(gates["minimum_oracle_rgb_rmse_gain_over_combined"]) != 0.1
        or float(gates["minimum_selected_rgb_rmse_gain_over_combined"]) != 0.05
        or float(
            gates["minimum_selected_rgb_rmse_gain_over_best_fixed_expert"]
        )
        != 0.03
        or float(
            gates["minimum_selected_mean_delta_e76_gain_over_combined"]
        )
        != 0.03
        or float(
            gates[
                "maximum_selected_to_combined_maximum_absolute_error_ratio"
            ]
        )
        != 1.25
        or float(gates["minimum_mean_weight_per_expert"]) != 0.1
        or not config["operator_fitting_allowed"]
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
    ):
        raise VelviaTwoExpertCVError("AO8B frozen contract mismatch")

    parent = _load_exact_json(root, config["parent_decision"])
    if parent.get("decision") != config["parent_decision"]["required_decision"]:
        raise VelviaTwoExpertCVError("AO8B parent decision mismatch")
    ao5 = _load_exact_json(root, config["inputs"]["ao5_config"])
    chart = config["inputs"]["chart_pairs"]
    palette = config["inputs"]["palette_pairs"]
    if (
        ao5.get("experiment_id")
        != "u5.r2ao5-combined-velvia-operator-v1"
        or ao5["inputs"]["chart_pairs"] != chart["path"]
        or ao5["inputs"]["chart_paired_u8_sha256"]
        != chart["paired_u8_sha256"]
        or int(ao5["inputs"]["chart_rows"]) != int(chart["rows"])
        or ao5["inputs"]["palette_pairs"] != palette["path"]
        or ao5["inputs"]["palette_asset_sha256"] != palette["asset_sha256"]
        or ao5["inputs"]["velvia_palette_paired_u8_sha256"]
        != palette["paired_u8_sha256"]
        or int(ao5["inputs"]["palette_rows"]) != int(palette["rows"])
        or ao5["fit"] != config["fit"]
    ):
        raise VelviaTwoExpertCVError("AO8B pair or fit contract mismatch")
    return ao5


def deterministic_domain_folds(
    row_count: int, fold_count: int, seed: int
) -> np.ndarray:
    """Assign rows to folds using a frozen source-row permutation."""

    if row_count < 2 * fold_count or fold_count < 2:
        raise VelviaTwoExpertCVError("insufficient rows for domain folds")
    permutation = np.random.default_rng(seed).permutation(row_count)
    folds = np.empty(row_count, dtype=np.int64)
    folds[permutation] = np.arange(row_count, dtype=np.int64) % fold_count
    if set(folds.tolist()) != set(range(fold_count)):
        raise VelviaTwoExpertCVError("domain fold assignment is incomplete")
    return folds


def _fit(source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]):
    fit = config["fit"]
    return fit_positive_film_response_operator(
        source,
        target,
        model=str(fit["model"]),
        identity_mixture=float(fit["identity_mixture"]),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(
            fit["maximum_function_evaluations"]
        ),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    )


def _nearest_distances(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    squared = np.sum(
        (query[:, None, :] - support[None, :, :]) ** 2, axis=2
    )
    return np.sqrt(np.min(squared, axis=1))


def _support_scale(support: np.ndarray, floor: float) -> float:
    squared = np.sum(
        (support[:, None, :] - support[None, :, :]) ** 2, axis=2
    )
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.min(squared, axis=1))
    return max(float(np.median(nearest)), floor)


def support_router_weights(
    query: np.ndarray,
    chart_support: np.ndarray,
    palette_support: np.ndarray,
    *,
    kind: str,
    temperature: float | None,
    scale_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return chart weights using source-colour support only."""

    chart_scale = _support_scale(chart_support, scale_floor)
    palette_scale = _support_scale(palette_support, scale_floor)
    chart_distance = _nearest_distances(query, chart_support) / chart_scale
    palette_distance = (
        _nearest_distances(query, palette_support) / palette_scale
    )
    if kind == "hard":
        weights = (chart_distance <= palette_distance).astype(np.float64)
    elif kind == "softmax":
        if temperature is None or not np.isfinite(temperature) or temperature <= 0:
            raise VelviaTwoExpertCVError("soft router temperature is invalid")
        logits = np.stack(
            (-chart_distance / temperature, -palette_distance / temperature),
            axis=1,
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights = probabilities[:, 0]
    else:
        raise VelviaTwoExpertCVError("unknown support router")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.any(weights > 1.0)
    ):
        raise VelviaTwoExpertCVError("router weights escaped [0, 1]")
    return weights, {
        "chart_support_scale": chart_scale,
        "palette_support_scale": palette_scale,
    }


def evaluate_colourwise_two_expert_cv(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Refit only on development folds and aggregate held-out predictions."""

    ao5 = validate_contract(root, config)
    pairs = load_combined_velvia_pairs(
        root / ao5["inputs"]["chart_pairs"],
        root / ao5["inputs"]["palette_pairs"],
        ao5,
    )
    chart_source, chart_target = pairs["velvia_chart"]
    palette_source, palette_target = pairs["velvia_palette"]
    source, target = pairs["combined"]
    nested = config["nested_evaluation"]
    fold_count = int(nested["fold_count"])
    chart_folds = deterministic_domain_folds(
        len(chart_source), fold_count, int(nested["chart_split_seed"])
    )
    palette_folds = deterministic_domain_folds(
        len(palette_source), fold_count, int(nested["palette_split_seed"])
    )
    combined_folds = np.concatenate((chart_folds, palette_folds))
    candidate_specs = config["router"]["candidate_bank"]
    prediction_names = [
        "identity",
        "fixed_chart_expert",
        "fixed_palette_expert",
        "combined_global_operator",
        "two_expert_oracle",
        *[row["candidate_id"] for row in candidate_specs],
    ]
    predictions = {
        name: np.empty_like(source, dtype=np.float64)
        for name in prediction_names
    }
    chart_weights = {
        row["candidate_id"]: np.empty(len(source), dtype=np.float64)
        for row in candidate_specs
    }
    fold_records: list[dict[str, Any]] = []
    offset = len(chart_source)
    for fold in range(fold_count):
        chart_dev = chart_folds != fold
        palette_dev = palette_folds != fold
        test = combined_folds == fold
        development = ~test
        chart_fit = _fit(
            chart_source[chart_dev], chart_target[chart_dev], config
        )
        palette_fit = _fit(
            palette_source[palette_dev], palette_target[palette_dev], config
        )
        combined_fit = _fit(
            source[development], target[development], config
        )
        query = source[test]
        chart_output = chart_fit.operator.apply(query)
        palette_output = palette_fit.operator.apply(query)
        combined_output = combined_fit.operator.apply(query)
        predictions["identity"][test] = query
        predictions["fixed_chart_expert"][test] = chart_output
        predictions["fixed_palette_expert"][test] = palette_output
        predictions["combined_global_operator"][test] = combined_output
        row_chart_error = np.sum(
            (chart_output - target[test]) ** 2, axis=1
        )
        row_palette_error = np.sum(
            (palette_output - target[test]) ** 2, axis=1
        )
        choose_chart = row_chart_error <= row_palette_error
        predictions["two_expert_oracle"][test] = np.where(
            choose_chart[:, None], chart_output, palette_output
        )
        support_records = {}
        for spec in candidate_specs:
            candidate_id = spec["candidate_id"]
            weights, scales = support_router_weights(
                query,
                chart_source[chart_dev],
                palette_source[palette_dev],
                kind=str(spec["kind"]),
                temperature=(
                    float(spec["temperature"])
                    if "temperature" in spec
                    else None
                ),
                scale_floor=float(config["router"]["support_scale_floor"]),
            )
            predictions[candidate_id][test] = (
                weights[:, None] * chart_output
                + (1.0 - weights[:, None]) * palette_output
            )
            chart_weights[candidate_id][test] = weights
            support_records[candidate_id] = {
                **scales,
                "held_out_mean_chart_weight": float(np.mean(weights)),
            }
        fold_records.append(
            {
                "fold": fold,
                "chart_development_rows": int(np.sum(chart_dev)),
                "palette_development_rows": int(np.sum(palette_dev)),
                "held_out_chart_rows": int(np.sum(chart_folds == fold)),
                "held_out_palette_rows": int(np.sum(palette_folds == fold)),
                "chart_fit": _fit_record(chart_fit),
                "palette_fit": _fit_record(palette_fit),
                "combined_fit": _fit_record(combined_fit),
                "support": support_records,
            }
        )

    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise VelviaTwoExpertCVError("held-out prediction coverage is incomplete")
    scores = {
        name: _metrics(values, target)
        for name, values in predictions.items()
    }
    for candidate_id, weights in chart_weights.items():
        scores[candidate_id]["mean_chart_weight"] = float(np.mean(weights))
        scores[candidate_id]["mean_palette_weight"] = float(
            np.mean(1.0 - weights)
        )
    candidate_ids = [row["candidate_id"] for row in candidate_specs]
    selected_id = min(
        candidate_ids,
        key=lambda name: (scores[name]["rgb_rmse"], name),
    )
    selected = scores[selected_id]
    combined = scores["combined_global_operator"]
    fixed_id = min(
        ("fixed_chart_expert", "fixed_palette_expert"),
        key=lambda name: (scores[name]["rgb_rmse"], name),
    )
    fixed = scores[fixed_id]
    oracle = scores["two_expert_oracle"]

    def gain(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            raise VelviaTwoExpertCVError("comparison denominator is invalid")
        return 1.0 - numerator / denominator

    gains = {
        "oracle_rgb_rmse_over_combined": gain(
            oracle["rgb_rmse"], combined["rgb_rmse"]
        ),
        "selected_rgb_rmse_over_combined": gain(
            selected["rgb_rmse"], combined["rgb_rmse"]
        ),
        "selected_rgb_rmse_over_best_fixed": gain(
            selected["rgb_rmse"], fixed["rgb_rmse"]
        ),
        "selected_mean_delta_e76_over_combined": gain(
            selected["mean_delta_e76"], combined["mean_delta_e76"]
        ),
        "selected_to_combined_maximum_absolute_error_ratio": (
            selected["rgb_maximum_absolute_error"]
            / combined["rgb_maximum_absolute_error"]
        ),
    }
    gates = config["gates"]
    all_fits_converged = all(
        row[name]["converged"]
        for row in fold_records
        for name in ("chart_fit", "palette_fit", "combined_fit")
    )
    oracle_pass = gains["oracle_rgb_rmse_over_combined"] >= float(
        gates["minimum_oracle_rgb_rmse_gain_over_combined"]
    )
    checks = {
        "oracle_product_value": oracle_pass,
        "selected_gain_over_combined": gains[
            "selected_rgb_rmse_over_combined"
        ]
        >= float(gates["minimum_selected_rgb_rmse_gain_over_combined"]),
        "selected_gain_over_best_fixed": gains[
            "selected_rgb_rmse_over_best_fixed"
        ]
        >= float(
            gates["minimum_selected_rgb_rmse_gain_over_best_fixed_expert"]
        ),
        "selected_perceptual_gain_over_combined": gains[
            "selected_mean_delta_e76_over_combined"
        ]
        >= float(
            gates["minimum_selected_mean_delta_e76_gain_over_combined"]
        ),
        "selected_maximum_error_guard": gains[
            "selected_to_combined_maximum_absolute_error_ratio"
        ]
        <= float(
            gates[
                "maximum_selected_to_combined_maximum_absolute_error_ratio"
            ]
        ),
        "both_experts_used": min(
            selected["mean_chart_weight"],
            selected["mean_palette_weight"],
        )
        >= float(gates["minimum_mean_weight_per_expert"]),
        "selected_in_cube": selected["raw_out_of_cube_fraction"]
        <= float(gates["maximum_raw_out_of_cube_fraction"]),
        "all_fold_fits_converged": all_fits_converged,
    }
    automatic_pass = all(checks.values())
    if automatic_pass:
        decision = config["decision_if_pass"]
    elif not oracle_pass:
        decision = config["decision_if_oracle_fails"]
    else:
        decision = config["decision_if_router_fails"]
    return {
        "schema": "neuro-film.u5.r2ao8b.velvia-colourwise-two-expert-cv-report.v1",
        "experiment_id": config["experiment_id"],
        "fold_assignments": {
            "chart": chart_folds.tolist(),
            "palette": palette_folds.tolist(),
        },
        "folds": fold_records,
        "scores": scores,
        "selected_candidate_id": selected_id,
        "best_fixed_expert_id": fixed_id,
        "gains": gains,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "VelviaTwoExpertCVError",
    "deterministic_domain_folds",
    "evaluate_colourwise_two_expert_cv",
    "support_router_weights",
    "validate_contract",
]
