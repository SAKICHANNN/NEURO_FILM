"""Group-held pairwise compatibility learning for explicit FiveK case operators."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval.fivek_casebank_oracle import _even_samples, _fit_one, _rmse
from src.eval.fivek_source_hard_retrieval import (
    _group_bootstrap,
    _operators,
    _rgb,
    _source_only_threshold,
    _standardized_distances,
    source_descriptor,
)
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


class FiveKPairwiseCompatibilityError(ValueError):
    """Raised when pairwise evidence violates its frozen information boundary."""


def combined_source_descriptor(
    rgb: np.ndarray, descriptor_spec: Mapping[str, Any]
) -> np.ndarray:
    """Return the fixed source-only descriptor used by the pairwise learner."""

    return np.concatenate(
        [
            source_descriptor(rgb, "spatial_photometric", descriptor_spec),
            source_descriptor(rgb, "tone_layout", descriptor_spec),
        ]
    ).astype(np.float64)


def _sign_stable_components(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


def fit_projection(
    features: np.ndarray, component_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic standardization and PCA on development sources only."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise FiveKPairwiseCompatibilityError("invalid descriptor matrix")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    standardized = (matrix - mean) / scale
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    count = min(int(component_count), len(vt))
    if count < 2:
        raise FiveKPairwiseCompatibilityError("insufficient PCA support")
    return mean, scale, _sign_stable_components(vt[:count])


def project_features(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    projected = ((matrix - mean) / scale) @ components.T
    if not np.all(np.isfinite(projected)):
        raise FiveKPairwiseCompatibilityError("non-finite projected descriptor")
    return projected


def pair_features(query: np.ndarray, case: np.ndarray) -> np.ndarray:
    """Build bounded-dimensional pair features; rows must already be aligned."""

    q = np.asarray(query, dtype=np.float64)
    c = np.asarray(case, dtype=np.float64)
    if q.shape != c.shape or q.ndim != 2:
        raise FiveKPairwiseCompatibilityError("pair feature shape mismatch")
    delta = q - c
    return np.concatenate((q, c, np.abs(delta), q * c), axis=1)


def _rank_targets(errors: np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def training_matrix(
    projected: np.ndarray,
    indices: np.ndarray,
    error_matrix: np.ndarray,
    *,
    shuffled_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create all off-diagonal query/case rows and within-query rank targets."""

    selected = np.asarray(indices, dtype=np.int64)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rng = np.random.default_rng(shuffled_seed)
    for query_index in selected:
        candidates = selected[selected != query_index]
        query = np.repeat(projected[query_index][None, :], len(candidates), axis=0)
        case = projected[candidates]
        target = _rank_targets(error_matrix[query_index, candidates])
        if shuffled_seed is not None:
            target = target[rng.permutation(len(target))]
        rows.append(pair_features(query, case))
        targets.append(target)
    return np.concatenate(rows), np.concatenate(targets)


def fit_ridge_ranker(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> dict[str, np.ndarray | float]:
    """Fit a deterministic closed-form ridge model."""

    matrix = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    normalized = (matrix - mean) / scale
    target_mean = float(np.mean(target))
    gram = normalized.T @ normalized
    rhs = normalized.T @ (target - target_mean)
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64), rhs
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": target_mean,
    }


def predict_ranker(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    return (
        (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficient"]) + float(model["intercept"])


def rank_queries(
    *,
    projected: np.ndarray,
    query_indices: np.ndarray,
    case_indices: np.ndarray,
    model: Mapping[str, Any],
) -> np.ndarray:
    chosen = []
    for query_index in np.asarray(query_indices, dtype=np.int64):
        query = np.repeat(
            projected[query_index][None, :], len(case_indices), axis=0
        )
        x = pair_features(query, projected[case_indices])
        scores = predict_ranker(model, x)
        order = np.lexsort((case_indices, scores))
        chosen.append(int(case_indices[int(order[0])]))
    return np.asarray(chosen, dtype=np.int64)


def _pooled_operator(
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    operator_config: Mapping[str, Any],
) -> TriangularLogitTransport:
    sources, targets = [], []
    per_image = int(operator_config["pooled_samples_per_image"])
    for index in np.asarray(indices, dtype=np.int64):
        sources.append(_even_samples(_rgb(rows[index]["source"]), per_image))
        targets.append(_even_samples(_rgb(rows[index]["target"]), per_image))
    source = np.concatenate(sources)
    target = np.concatenate(targets)
    _, operator, _ = _fit_one(
        source[:, None, :],
        target[:, None, :],
        {**operator_config, "fit_samples_per_image": len(source)},
    )
    return operator


def _pooled_errors(
    rows: Sequence[Mapping[str, Any]],
    query_indices: np.ndarray,
    operator: TriangularLogitTransport,
    samples: int,
) -> np.ndarray:
    values = []
    for index in np.asarray(query_indices, dtype=np.int64):
        source = _even_samples(_rgb(rows[index]["source"]), samples)
        target = _even_samples(_rgb(rows[index]["target"]), samples)
        values.append(_rmse(operator.apply(source), target))
    return np.asarray(values, dtype=np.float64)


def _deterministic_random_cases(
    ids: Sequence[str], query_ids: Sequence[str], case_indices: np.ndarray, seed: int
) -> np.ndarray:
    output = []
    for query_id in query_ids:
        digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
        output.append(int(case_indices[int.from_bytes(digest[:8], "big") % len(case_indices)]))
    return np.asarray(output, dtype=np.int64)


def _model_identity(
    projection: tuple[np.ndarray, np.ndarray, np.ndarray],
    model: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for value in (*projection, model["mean"], model["scale"], model["coefficient"]):
        digest.update(np.asarray(value, dtype="<f8").tobytes())
    digest.update(np.asarray([model["intercept"]], dtype="<f8").tobytes())
    return digest.hexdigest()


def _metrics(
    *,
    baseline: np.ndarray,
    selected: np.ndarray,
    nearest: np.ndarray,
    oracle: np.ndarray,
    random: np.ndarray,
    shuffled: np.ndarray,
    groups: np.ndarray,
    selected_ids: list[str],
    fallback: np.ndarray,
    gates: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    bootstrap = _group_bootstrap(
        baseline,
        selected,
        groups,
        seed=int(selector["bootstrap_seed"]),
        repetitions=int(selector["bootstrap_repetitions"]),
    )
    counts = Counter(
        case_id for case_id, is_fallback in zip(selected_ids, fallback) if not is_fallback
    )
    base_mean = float(np.mean(baseline))
    selected_mean = float(np.mean(selected))
    nearest_mean = float(np.mean(nearest))
    oracle_mean = float(np.mean(oracle))
    shuffled_mean = float(np.mean(shuffled))
    metrics = {
        "mean_improvement_over_global": (base_mean - selected_mean) / max(base_mean, 1e-12),
        "win_fraction_over_global": float(np.mean(selected < baseline)),
        "p95_ratio_to_global": float(np.quantile(selected, 0.95)) / max(float(np.quantile(baseline, 0.95)), 1e-12),
        "worst_ratio_to_global": float(np.max(selected)) / max(float(np.max(baseline)), 1e-12),
        "oracle_gap_closure": (base_mean - selected_mean) / max(base_mean - oracle_mean, 1e-12),
        "mean_improvement_over_nearest": (nearest_mean - selected_mean) / max(nearest_mean, 1e-12),
        "win_fraction_over_nearest": float(np.mean(selected < nearest)),
        "mean_improvement_over_random_case": (float(np.mean(random)) - selected_mean) / max(float(np.mean(random)), 1e-12),
        "mean_improvement_over_shuffled_ranker": (shuffled_mean - selected_mean) / max(shuffled_mean, 1e-12),
        "group_bootstrap_improvement_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "fallback_fraction": float(np.mean(fallback)),
        "distinct_selected_cases": len(counts),
        "maximum_selected_case_share": max(counts.values()) / len(selected) if counts else 0.0,
    }
    checks = {
        "mean": metrics["mean_improvement_over_global"] >= gates["minimum_mean_improvement_over_global"],
        "wins": metrics["win_fraction_over_global"] >= gates["minimum_win_fraction_over_global"],
        "p95": metrics["p95_ratio_to_global"] <= gates["maximum_p95_ratio_to_global"],
        "worst": metrics["worst_ratio_to_global"] <= gates["maximum_worst_ratio_to_global"],
        "oracle_gap": metrics["oracle_gap_closure"] >= gates["minimum_oracle_gap_closure"],
        "nearest_mean": metrics["mean_improvement_over_nearest"] >= gates["minimum_mean_improvement_over_nearest"],
        "nearest_wins": metrics["win_fraction_over_nearest"] >= gates["minimum_win_fraction_over_nearest"],
        "random": metrics["mean_improvement_over_random_case"] >= gates["minimum_mean_improvement_over_random_case"],
        "shuffle": metrics["mean_improvement_over_shuffled_ranker"] >= gates["minimum_mean_improvement_over_shuffled_ranker"],
        "bootstrap": metrics["group_bootstrap_improvement_ci95"][0] > gates["minimum_bootstrap_lower_improvement"],
        "fallback": metrics["fallback_fraction"] <= gates["maximum_fallback_fraction"],
        "support": metrics["distinct_selected_cases"] >= gates["minimum_distinct_selected_cases"],
        "concentration": metrics["maximum_selected_case_share"] <= gates["maximum_selected_case_share"],
    }
    return {"metrics": metrics, "gates": checks, "automatic_pass": all(checks.values()), "selected_case_counts": dict(sorted(counts.items()))}


def _group_partition(groups: np.ndarray, modulus: int, validation_bucket: int) -> tuple[np.ndarray, np.ndarray]:
    is_validation = np.asarray(
        [hashlib.sha256(str(group).encode("utf-8")).digest()[0] % modulus == validation_bucket for group in groups],
        dtype=bool,
    )
    fit = np.flatnonzero(~is_validation)
    validation = np.flatnonzero(is_validation)
    if len(set(groups[fit])) < 4 or len(set(groups[validation])) < 3:
        raise FiveKPairwiseCompatibilityError("insufficient group partition support")
    return fit, validation


def evaluate_development(
    *,
    rows: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    nearest_report: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    selector_spec: Mapping[str, Any],
    operator_config: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    ids = list(prepared["ids"])
    groups = np.asarray(prepared["groups"], dtype=object)
    features = np.stack([combined_source_descriptor(row["source"], descriptor_spec) for row in ordered])
    fit, validation = _group_partition(groups, int(model_spec["group_bucket_modulus"]), int(model_spec["validation_bucket"]))
    projection = fit_projection(features[fit], int(model_spec["pca_components"]))
    projected = project_features(features, *projection)
    x, y = training_matrix(projected, fit, np.asarray(prepared["error_matrix"]))
    model = fit_ridge_ranker(x, y, float(model_spec["ridge_alpha"]))
    shuffled_x, shuffled_y = training_matrix(projected, fit, np.asarray(prepared["error_matrix"]), shuffled_seed=int(model_spec["shuffled_label_seed"]))
    shuffled_model = fit_ridge_ranker(shuffled_x, shuffled_y, float(model_spec["ridge_alpha"]))
    selected_case = rank_queries(projected=projected, query_indices=validation, case_indices=fit, model=model)
    shuffled_case = rank_queries(projected=projected, query_indices=validation, case_indices=fit, model=shuffled_model)
    distances, _, _ = _standardized_distances(projected[fit], projected[validation])
    nearest_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(projected[fit], groups[fit], float(selector_spec["ood_distance_quantile"]))
    fallback = nearest_distance > threshold
    pooled = _pooled_operator(ordered, fit, operator_config)
    baseline = _pooled_errors(
        ordered,
        validation,
        pooled,
        int(model_spec["development_samples_per_image"]),
    )
    matrix = np.asarray(prepared["error_matrix"])
    selected = matrix[validation, selected_case]
    shuffled = matrix[validation, shuffled_case]
    selected = np.where(fallback, baseline, selected)
    shuffled = np.where(fallback, baseline, shuffled)
    oracle = np.min(matrix[np.ix_(validation, fit)], axis=1)
    random_case = _deterministic_random_cases(ids, [ids[i] for i in validation], fit, int(selector_spec["random_case_seed"]))
    random = matrix[validation, random_case]
    nearest_rows = {str(row["pair_id"]): row for row in nearest_report["rows"]}
    nearest = np.asarray([float(nearest_rows[ids[i]]["selected_rmse"]) for i in validation])
    selected_ids = [ids[i] for i in selected_case]
    result = _metrics(
        baseline=baseline, selected=selected, nearest=nearest, oracle=oracle,
        random=random, shuffled=shuffled, groups=groups[validation],
        selected_ids=selected_ids, fallback=fallback, gates=gates, selector=selector_spec,
    )
    result.update({
        "fit_rows": len(fit), "validation_rows": len(validation),
        "fit_groups": sorted(set(map(str, groups[fit]))),
        "validation_groups": sorted(set(map(str, groups[validation]))),
        "model_sha256": _model_identity(projection, model),
        "rows": [
            {
                "pair_id": ids[index], "group": str(groups[index]),
                "selected_case_id": selected_ids[pos], "fallback": bool(fallback[pos]),
                "distance": float(nearest_distance[pos]), "threshold": threshold,
                "global_rmse": float(baseline[pos]), "selected_rmse": float(selected[pos]),
                "nearest_rmse": float(nearest[pos]), "oracle_rmse": float(oracle[pos]),
            }
            for pos, index in enumerate(validation)
        ],
    })
    return result


def evaluate_confirmation(
    *,
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    oracle_report: Mapping[str, Any],
    nearest_report: Mapping[str, Any],
    descriptor_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    selector_spec: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    development = sorted(development_rows, key=lambda row: str(row["pair_id"]))
    confirmation = sorted(confirmation_rows, key=lambda row: str(row["pair_id"]))
    ids = [str(row["pair_id"]) for row in development]
    groups = np.asarray([str(row["group"]) for row in development], dtype=object)
    all_rows = development + confirmation
    features = np.stack([combined_source_descriptor(row["source"], descriptor_spec) for row in all_rows])
    fit = np.arange(len(development), dtype=np.int64)
    query = np.arange(len(development), len(all_rows), dtype=np.int64)
    projection = fit_projection(features[fit], int(model_spec["pca_components"]))
    projected = project_features(features, *projection)
    operators = _operators(oracle_report)
    development_samples = int(model_spec["development_samples_per_image"])
    confirmation_samples = int(model_spec["confirmation_samples_per_image"])
    error_matrix = np.empty((len(development), len(development)), dtype=np.float64)
    for query_index, row in enumerate(development):
        source = _even_samples(_rgb(row["source"]), development_samples)
        target = _even_samples(_rgb(row["target"]), development_samples)
        for case_index, operator in enumerate(operators):
            error_matrix[query_index, case_index] = _rmse(operator.apply(source), target)
    x, y = training_matrix(projected[: len(development)], fit, error_matrix)
    model = fit_ridge_ranker(x, y, float(model_spec["ridge_alpha"]))
    sx, sy = training_matrix(projected[: len(development)], fit, error_matrix, shuffled_seed=int(model_spec["shuffled_label_seed"]))
    shuffled_model = fit_ridge_ranker(sx, sy, float(model_spec["ridge_alpha"]))
    selected_case = rank_queries(projected=projected, query_indices=query, case_indices=fit, model=model)
    shuffled_case = rank_queries(projected=projected, query_indices=query, case_indices=fit, model=shuffled_model)
    distances, _, _ = _standardized_distances(projected[fit], projected[query])
    nearest_distance = np.min(distances, axis=1)
    threshold = _source_only_threshold(projected[fit], groups, float(selector_spec["ood_distance_quantile"]))
    fallback = nearest_distance > threshold
    pooled = TriangularLogitTransport(np.asarray(oracle_report["pooled_operator"]["parameters"], dtype=np.float64), dose=float(oracle_report["pooled_operator"]["dose"]))
    oracle_rows = {str(row["pair_id"]): row for row in oracle_report["rows"]}
    nearest_rows = {str(row["pair_id"]): row for row in nearest_report["rows"]}
    baseline, selected, nearest, oracle, random, shuffled, selected_ids, output_rows = [], [], [], [], [], [], [], []
    for pos, row in enumerate(confirmation):
        pair_id = str(row["pair_id"])
        source = _even_samples(_rgb(row["source"]), confirmation_samples)
        target = _even_samples(_rgb(row["target"]), confirmation_samples)
        chosen = int(selected_case[pos])
        shuffled_chosen = int(shuffled_case[pos])
        evidence = oracle_rows[pair_id]
        base_error = float(evidence["global_rmse"])
        selected_error = base_error if fallback[pos] else _rmse(operators[chosen].apply(source), target)
        shuffled_error = base_error if fallback[pos] else _rmse(operators[shuffled_chosen].apply(source), target)
        baseline.append(base_error); selected.append(selected_error); shuffled.append(shuffled_error)
        nearest.append(float(nearest_rows[pair_id]["selected_rmse"]))
        oracle.append(float(evidence["case_oracle_rmse"])); random.append(float(evidence["random_case_rmse"]))
        selected_ids.append(ids[chosen])
        output_rows.append({
            "pair_id": pair_id, "group": str(row["group"]), "selected_case_id": ids[chosen],
            "fallback": bool(fallback[pos]), "distance": float(nearest_distance[pos]), "threshold": threshold,
            "global_rmse": base_error, "selected_rmse": selected_error,
            "nearest_rmse": float(nearest_rows[pair_id]["selected_rmse"]),
            "oracle_rmse": float(evidence["case_oracle_rmse"]),
        })
    result = _metrics(
        baseline=np.asarray(baseline), selected=np.asarray(selected), nearest=np.asarray(nearest),
        oracle=np.asarray(oracle), random=np.asarray(random), shuffled=np.asarray(shuffled),
        groups=np.asarray([str(row["group"]) for row in confirmation], dtype=object),
        selected_ids=selected_ids, fallback=fallback, gates=gates, selector=selector_spec,
    )
    result.update({"model_sha256": _model_identity(projection, model), "rows": output_rows})
    return result


__all__ = [
    "FiveKPairwiseCompatibilityError", "combined_source_descriptor",
    "evaluate_confirmation", "evaluate_development", "fit_projection",
    "fit_ridge_ranker", "pair_features", "predict_ranker", "project_features",
    "rank_queries", "training_matrix",
]
