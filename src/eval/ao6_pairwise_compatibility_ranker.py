"""Group-crossfit pairwise compatibility ranker for hard AO6 cases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge

from src.eval.ao6_case_operator_oracle import _apply_context, _canonical_sha256, _rmse
from src.eval.ao6_content_hard_retrieval import _extract_features, _source_view
from src.eval.ao6_global_lut_distillation import (
    _aligned_sample,
    _decode_target,
    _load_exact_json,
    _load_source,
)
from src.eval.ao6_tone_layout_hard_retrieval import (
    _extract_tone_layout_features,
    validate_contract as validate_bm3_contract,
)
from src.eval.global_frontier import sha256_file
from src.film_physics.display_look import build_density_source_context_row_staged


SCHEMA = "neuro_film.u5_r2bm4_ao6_pairwise_compatibility_ranker_report.v1"


class AO6PairwiseCompatibilityRankerError(RuntimeError):
    """Raised when BM4 evidence, splits or information flow drift."""


def _pair_feature(
    query: int,
    case: int,
    semantic: np.ndarray,
    tone: np.ndarray,
) -> np.ndarray:
    cosine = float(semantic[query] @ semantic[case])
    return np.concatenate(([cosine], np.abs(tone[query] - tone[case]))).astype(
        np.float64
    )


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[Ridge, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.maximum(np.std(x, axis=0), 1e-6)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit((x - mean) / std, y)
    return model, mean, std


def _predict(
    model: Ridge, mean: np.ndarray, std: np.ndarray, x: np.ndarray
) -> np.ndarray:
    return np.asarray(model.predict((x - mean) / std), dtype=np.float64)


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    features = config["features"]
    ranker = config["ranker"]
    if (
        config.get("schema") != "neuro_film.u5_r2bm4_ao6_pairwise_compatibility_ranker.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or not config.get("training_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("latent_mode_claim_allowed")
        or config.get("product_integration_allowed")
        or features.get("source_chroma_used")
        or features.get("operator_signature_used_at_inference")
        or features.get("target_or_output_used_at_inference")
        or ranker.get("dense_blending_allowed")
        or ranker.get("final_rgb_prediction_allowed")
        or int(features.get("dimension", -1)) != 75
    ):
        raise AO6PairwiseCompatibilityRankerError("BM4 frozen contract drift")
    parent = config["parent"]
    decision = _load_exact_json(root, parent["decision"], parent["decision_sha256"])
    if decision.get("decision") != parent["required_decision"]:
        raise AO6PairwiseCompatibilityRankerError("BM3 decision does not open BM4")
    bm3_config = _load_exact_json(
        root, decision["contract"]["path"], decision["contract"]["sha256"]
    )
    validated = validate_bm3_contract(root, bm3_config)
    bm2_decision = _load_exact_json(
        root,
        bm3_config["parent"]["decision"],
        bm3_config["parent"]["decision_sha256"],
    )
    bm2_config = _load_exact_json(
        root,
        bm2_decision["contract"]["path"],
        bm2_decision["contract"]["sha256"],
    )
    bm1_report = _load_exact_json(
        root, parent["bm1_report"], parent["bm1_report_sha256"]
    )
    if not bm1_report.get("automatic_pass"):
        raise AO6PairwiseCompatibilityRankerError("BM1 Oracle drift")
    return {
        **validated,
        "bm1_report": bm1_report,
        "bm2_config": bm2_config,
        "bm3_config": bm3_config,
    }


def _training_rows(
    queries: list[str],
    cases: list[str],
    indices: Mapping[str, int],
    semantic: np.ndarray,
    tone: np.ndarray,
    errors: Mapping[str, Mapping[str, float]],
    global_errors: Mapping[str, float],
    labels: Mapping[tuple[str, str], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for query in queries:
        for case in cases:
            if query == case:
                continue
            x.append(_pair_feature(indices[query], indices[case], semantic, tone))
            y.append(
                labels[(query, case)]
                if labels is not None
                else np.log(errors[query][case] / global_errors[query])
            )
    return np.stack(x), np.asarray(y, dtype=np.float64)


def _rank_query(
    query: str,
    cases: list[str],
    indices: Mapping[str, int],
    semantic: np.ndarray,
    tone: np.ndarray,
    model: Ridge,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[str, float, list[float]]:
    candidates = [case for case in cases if case != query]
    x = np.stack(
        [_pair_feature(indices[query], indices[case], semantic, tone) for case in candidates]
    )
    scores = _predict(model, mean, std, x)
    order = sorted(range(len(candidates)), key=lambda i: (float(scores[i]), candidates[i]))
    margin = float(scores[order[1]] - scores[order[0]]) if len(order) > 1 else 0.0
    return candidates[order[0]], margin, [float(value) for value in scores]


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise AO6PairwiseCompatibilityRankerError("BM4 requires OMP_NUM_THREADS=1")
    if output_dir.exists():
        raise FileExistsError("BM4 output is create-only")
    output_dir.mkdir(parents=True)
    validated = validate_contract(root, config)
    source_rows = validated["source_rows"]
    target_rows = validated["target_rows"]
    payload = validated["display_payload"]
    bm1 = validated["bm1_report"]
    case_protocol = validated["case_protocol"]
    source_ids = sorted(source_rows)
    indices = {source_id: index for index, source_id in enumerate(source_ids)}
    contexts, samples, views = {}, {}, []
    for source_id in source_ids:
        source = _load_source(root, source_rows[source_id])
        contexts[source_id] = build_density_source_context_row_staged(
            payload, source, tile_rows=int(case_protocol["context_tile_rows"])
        )
        spec = validated["bm2_config"]["feature_model"]
        views.append(
            _source_view(
                source,
                resize_short_edge=int(spec["resize_short_edge"]),
                image_size=int(spec["image_size"]),
            )
        )
        target_row = target_rows[source_id]
        target = _decode_target(
            validated["target_root"] / target_row["output"], target_row["output_sha256"]
        )
        source_sample, target_sample = _aligned_sample(
            source, target, int(case_protocol["evaluation_samples_per_source"])
        )
        side = int(round(np.sqrt(len(source_sample))))
        samples[source_id] = (
            source_sample.reshape(side, side, 3),
            target_sample.reshape(side, side, 3),
        )
        del source, target
    semantic = _extract_features(
        root=root, views=views, config=validated["bm2_config"]
    ).astype(np.float64)
    semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
    tone = _extract_tone_layout_features(
        root=root, views=views, config=validated["bm3_config"]
    ).astype(np.float64)
    feature_sha = hashlib.sha256(
        semantic.astype("<f4").tobytes() + tone.astype("<f4").tobytes()
    ).hexdigest()
    folds = {int(row["held_fold"]): row for row in bm1["fold_models"]}
    bm1_rows = {row["source_id"]: row for row in bm1["rows"]}
    errors: dict[str, dict[str, float]] = {}
    style: dict[str, dict[str, float]] = {}
    boundary: dict[str, dict[str, float]] = {}
    epsilon = float(case_protocol["boundary_epsilon_encoded_srgb"])
    for query in source_ids:
        source_sample, target_sample = samples[query]
        target_style = _rmse(target_sample, source_sample)
        target_boundary = np.any(
            (target_sample <= epsilon) | (target_sample >= 1.0 - epsilon), axis=-1
        )
        errors[query], style[query], boundary[query] = {}, {}, {}
        for case in source_ids:
            output = _apply_context(payload, contexts[case], source_sample)
            errors[query][case] = _rmse(output, target_sample)
            style[query][case] = _rmse(output, source_sample) / target_style
            output_boundary = np.any(
                (output <= epsilon) | (output >= 1.0 - epsilon), axis=-1
            )
            boundary[query][case] = float(np.mean(output_boundary & ~target_boundary))

    alpha_grid = [float(value) for value in config["ranker"]["alpha_grid"]]
    fold_models, rows = [], []
    shuffled_rows = []
    randomizer = random.Random(int(config["controls"]["shuffled_seed"]))
    for held_fold, fold in sorted(folds.items()):
        development = list(fold["development_case_ids"])
        held = list(fold["held_source_ids"])
        medoid = str(fold["global_medoid_case_id"])
        global_errors = {query: errors[query][medoid] for query in source_ids}
        inner_scores: dict[float, list[float]] = {alpha: [] for alpha in alpha_grid}
        for inner_held in development:
            inner_dev = [item for item in development if item != inner_held]
            x_train, y_train = _training_rows(
                inner_dev, inner_dev, indices, semantic, tone, errors, global_errors
            )
            for alpha in alpha_grid:
                model, mean, std = _fit_ridge(x_train, y_train, alpha)
                selected, _, _ = _rank_query(
                    inner_held, inner_dev, indices, semantic, tone, model, mean, std
                )
                inner_scores[alpha].append(errors[inner_held][selected] / global_errors[inner_held])
        alpha = min(
            alpha_grid,
            key=lambda value: (float(np.mean(inner_scores[value])), -value),
        )
        safe_margins = []
        for inner_held in development:
            inner_dev = [item for item in development if item != inner_held]
            x_train, y_train = _training_rows(
                inner_dev, inner_dev, indices, semantic, tone, errors, global_errors
            )
            model, mean, std = _fit_ridge(x_train, y_train, alpha)
            selected, margin, _ = _rank_query(
                inner_held, inner_dev, indices, semantic, tone, model, mean, std
            )
            if errors[inner_held][selected] <= global_errors[inner_held]:
                safe_margins.append(margin)
        minimum_safe = int(config["fallback"]["minimum_safe_inner_queries"])
        threshold = (
            float(np.quantile(safe_margins, 0.1, method="linear"))
            if len(safe_margins) >= minimum_safe
            else float(np.finfo(np.float64).max)
        )
        x_train, y_train = _training_rows(
            development, development, indices, semantic, tone, errors, global_errors
        )
        model, mean, std = _fit_ridge(x_train, y_train, alpha)
        label_map: dict[tuple[str, str], float] = {}
        for query in development:
            cases = [case for case in development if case != query]
            values = [np.log(errors[query][case] / global_errors[query]) for case in cases]
            randomizer.shuffle(values)
            label_map.update({(query, case): value for case, value in zip(cases, values, strict=True)})
        _, shuffled_y = _training_rows(
            development,
            development,
            indices,
            semantic,
            tone,
            errors,
            global_errors,
            label_map,
        )
        shuffled_model, shuffled_mean, shuffled_std = _fit_ridge(x_train, shuffled_y, alpha)
        fold_models.append(
            {
                "held_fold": held_fold,
                "selected_alpha": alpha,
                "inner_mean_ratios": {str(key): float(np.mean(value)) for key, value in inner_scores.items()},
                "safe_inner_queries": len(safe_margins),
                "confidence_margin_threshold": threshold,
                "coefficient_sha256": hashlib.sha256(
                    np.asarray(model.coef_, dtype="<f8").tobytes()
                    + np.asarray([model.intercept_], dtype="<f8").tobytes()
                ).hexdigest(),
                "normalization_sha256": hashlib.sha256(
                    np.asarray(mean, dtype="<f8").tobytes()
                    + np.asarray(std, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
        for query in held:
            selected, margin, _ = _rank_query(
                query, development, indices, semantic, tone, model, mean, std
            )
            authorized = margin >= threshold
            retrieved = selected if authorized else medoid
            shuffled_selected, shuffled_margin, _ = _rank_query(
                query,
                development,
                indices,
                semantic,
                tone,
                shuffled_model,
                shuffled_mean,
                shuffled_std,
            )
            shuffled_retrieved = shuffled_selected if shuffled_margin >= threshold else medoid
            oracle = min(development, key=lambda case: (errors[query][case], case))
            parent = bm1_rows[query]
            if abs(errors[query][medoid] - parent["global_medoid_target_rgb_rmse"]) > 1e-12:
                raise AO6PairwiseCompatibilityRankerError("BM1 replay drift")
            rows.append(
                {
                    "source_id": query,
                    "held_fold": held_fold,
                    "global_medoid_case_id": medoid,
                    "predicted_case_id": selected,
                    "confidence_margin": margin,
                    "fallback": not authorized,
                    "retrieved_case_id": retrieved,
                    "oracle_case_id": oracle,
                    "global_target_rgb_rmse": errors[query][medoid],
                    "retrieval_target_rgb_rmse": errors[query][retrieved],
                    "retrieval_vs_global_rmse_ratio": errors[query][retrieved] / errors[query][medoid],
                    "style_retention_ratio": style[query][retrieved],
                    "new_boundary_fraction_vs_ao6": boundary[query][retrieved],
                }
            )
            shuffled_rows.append(
                errors[query][shuffled_retrieved] / errors[query][medoid]
            )
    ratios = np.asarray([row["retrieval_vs_global_rmse_ratio"] for row in rows])
    reduction = float(1.0 - np.median(ratios))
    shuffled_reduction = float(1.0 - np.median(shuffled_rows))
    fold_reductions = {
        str(fold): float(
            1.0
            - np.median(
                [row["retrieval_vs_global_rmse_ratio"] for row in rows if row["held_fold"] == fold]
            )
        )
        for fold in folds
    }
    aggregate = {
        "source_count": len(rows),
        "feature_sha256": feature_sha,
        "median_retrieval_rmse_reduction_vs_global_medoid": reduction,
        "fraction_of_bm1_oracle_gain_recovered": reduction / bm1["aggregate"]["median_case_rmse_reduction_vs_global_medoid"],
        "worst_retrieval_vs_global_rmse_ratio": float(np.max(ratios)),
        "sources_retrieval_beats_global": int(np.sum(ratios < 1.0)),
        "improved_outer_folds": int(sum(value > 0.0 for value in fold_reductions.values())),
        "nonfallback_sources": int(sum(not row["fallback"] for row in rows)),
        "shuffled_reduction": shuffled_reduction,
        "retrieval_reduction_advantage_over_shuffled": reduction - shuffled_reduction,
        "median_style_retention_ratio": float(np.median([row["style_retention_ratio"] for row in rows])),
        "maximum_new_boundary_fraction_vs_ao6": float(max(row["new_boundary_fraction_vs_ao6"] for row in rows)),
        "oracle_exact_match_count": int(sum(row["retrieved_case_id"] == row["oracle_case_id"] for row in rows)),
        "fold_reductions": fold_reductions,
    }
    gates = config["gates"]
    checks = {
        "value": reduction >= gates["minimum_median_retrieval_rmse_reduction_vs_global_medoid"],
        "oracle_recovery": aggregate["fraction_of_bm1_oracle_gain_recovered"] >= gates["minimum_fraction_of_bm1_oracle_gain_recovered"],
        "tail": aggregate["worst_retrieval_vs_global_rmse_ratio"] <= gates["maximum_worst_retrieval_vs_global_rmse_ratio"],
        "source_support": aggregate["sources_retrieval_beats_global"] >= gates["minimum_sources_retrieval_beats_global"],
        "fold_support": aggregate["improved_outer_folds"] >= gates["minimum_improved_outer_folds"],
        "nonfallback": aggregate["nonfallback_sources"] >= gates["minimum_nonfallback_sources"],
        "shuffle": aggregate["retrieval_reduction_advantage_over_shuffled"] >= gates["minimum_retrieval_reduction_advantage_over_shuffled"],
        "style": gates["minimum_median_style_retention_ratio"] <= aggregate["median_style_retention_ratio"] <= gates["maximum_median_style_retention_ratio"],
        "boundary": aggregate["maximum_new_boundary_fraction_vs_ao6"] <= gates["maximum_new_boundary_fraction_vs_ao6"],
    }
    core = {
        "schema": SCHEMA,
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "feature_sha256": feature_sha,
        "models": fold_models,
        "rows": rows,
        "aggregate": aggregate,
        "gates": checks,
        "automatic_pass": bool(all(checks.values())),
        "fresh_confirmation_opened": bool(all(checks.values())),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**core, "stable_evidence_id": _canonical_sha256(core)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["AO6PairwiseCompatibilityRankerError", "run_audit", "validate_contract"]
