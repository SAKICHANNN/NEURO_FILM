"""Transparent case routing for paired-supervised REPID operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.repid_paired_style_latent import (
    encode_pca,
    predict_factorized,
    train_factorized_models,
)


def _standardize_fit(values: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid fit feature matrix")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.maximum(scale, float(epsilon))
    standardized = (matrix - mean) / scale
    norms = np.linalg.norm(standardized, axis=1)
    if np.any(norms <= float(epsilon)):
        raise ValueError("zero-norm standardized fit feature")
    return standardized / norms[:, None], mean, scale


def _standardize_query(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    matrix = (np.asarray(values, dtype=np.float64) - mean) / scale
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(norms <= float(epsilon)):
        raise ValueError("invalid standardized query feature")
    return matrix / norms[:, None]


def _top_k_weights(scores: np.ndarray, top_k: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    if top_k <= 0 or top_k > values.size or temperature <= 0.0:
        raise ValueError("invalid top-k or temperature")
    order = np.lexsort((np.arange(values.size), -values))[:top_k]
    logits = values[order] / float(temperature)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return order, weights


def _medoid_index(latent: np.ndarray) -> int:
    values = np.asarray(latent, dtype=np.float64)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    totals = np.sum(distances, axis=1)
    return int(np.argmin(totals))


def train_case_bank(
    after_features: np.ndarray,
    paired_deltas: np.ndarray,
    operator_parameters: np.ndarray,
    *,
    latent_spec: Mapping[str, Any],
    retrieval_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the frozen REPID13 factorization and transparent case bank."""

    features = np.asarray(after_features, dtype=np.float64)
    deltas = np.asarray(paired_deltas, dtype=np.float64)
    parameters = np.asarray(operator_parameters, dtype=np.float64)
    if features.shape[0] != deltas.shape[0] or features.shape[0] != parameters.shape[0]:
        raise ValueError("case-bank row mismatch")
    factorized = train_factorized_models(
        features,
        deltas,
        parameters,
        component_count=int(latent_spec["principal_components"]),
        student_alpha=float(latent_spec["student_ridge_alpha"]),
        operator_alpha=float(latent_spec["operator_ridge_alpha"]),
        cyclic_shift=int(retrieval_spec["cyclic_wrong_label_shift"]),
    )
    normalized, mean, scale = _standardize_fit(features, 1e-8)
    latent = encode_pca(factorized["pca"], deltas)
    return {
        "factorized": factorized,
        "fit_features_normalized": normalized,
        "feature_mean": mean,
        "feature_scale": scale,
        "fit_latent": latent,
        "fit_parameters": parameters,
        "cyclic_parameters": np.roll(
            parameters,
            int(retrieval_spec["cyclic_wrong_label_shift"]),
            axis=0,
        ),
        "medoid_index": _medoid_index(latent),
    }


def predict_case_routes(
    model: Mapping[str, Any],
    after_features: np.ndarray,
    retrieval_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return frozen after-only top-3, controls, neighbors and weights."""

    queries = _standardize_query(
        after_features,
        np.asarray(model["feature_mean"]),
        np.asarray(model["feature_scale"]),
        1e-8,
    )
    fit_features = np.asarray(model["fit_features_normalized"])
    parameters = np.asarray(model["fit_parameters"])
    cyclic = np.asarray(model["cyclic_parameters"])
    top_k = int(retrieval_spec["primary_top_k"])
    temperature = float(retrieval_spec["softmax_temperature"])
    candidate: list[np.ndarray] = []
    top1: list[np.ndarray] = []
    wrong: list[np.ndarray] = []
    neighbors: list[list[int]] = []
    weights_out: list[list[float]] = []
    for query in queries:
        indices, weights = _top_k_weights(fit_features @ query, top_k, temperature)
        candidate.append(weights @ parameters[indices])
        top1.append(parameters[indices[0]])
        wrong.append(weights @ cyclic[indices])
        neighbors.append(indices.tolist())
        weights_out.append(weights.tolist())
    student = predict_factorized(model["factorized"], np.asarray(after_features))
    medoid = parameters[int(model["medoid_index"])]
    global_parameters = np.asarray(model["factorized"]["global_parameters"])
    row_count = queries.shape[0]
    return {
        "candidate_parameters": np.stack(candidate),
        "top1_parameters": np.stack(top1),
        "medoid_parameters": np.repeat(medoid[None, :], row_count, axis=0),
        "global_parameters": np.repeat(global_parameters[None, :], row_count, axis=0),
        "repid13_student_parameters": student["candidate_parameters"],
        "cyclic_parameters": np.stack(wrong),
        "neighbor_indices": neighbors,
        "neighbor_weights": weights_out,
    }


def predict_latent_oracle(
    model: Mapping[str, Any],
    actual_paired_deltas: np.ndarray,
    retrieval_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Diagnostic top-3 retrieval from paired latents, evaluated post-freeze."""

    queries = encode_pca(model["factorized"]["pca"], actual_paired_deltas)
    fit_latent = np.asarray(model["fit_latent"])
    parameters = np.asarray(model["fit_parameters"])
    top_k = int(retrieval_spec["primary_top_k"])
    temperature = float(retrieval_spec["softmax_temperature"])
    output: list[np.ndarray] = []
    neighbors: list[list[int]] = []
    for query in queries:
        distances = np.linalg.norm(fit_latent - query[None, :], axis=1)
        indices, weights = _top_k_weights(-distances, top_k, temperature)
        output.append(weights @ parameters[indices])
        neighbors.append(indices.tolist())
    return {"parameters": np.stack(output), "neighbor_indices": neighbors}


__all__ = ["predict_case_routes", "predict_latent_oracle", "train_case_bank"]
