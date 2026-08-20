"""Paired-supervised, after-only explicit-operator identification for REPID."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.eval.filmmatch_source_conditioned_operator import (
    fit_ridge,
    predict_ridge,
    source_descriptor,
)
from src.eval.repid_reference_operator_predictor import resize_srgb


def spatial_descriptor(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Return a fixed global plus row-major spatial colour descriptor."""

    values = np.asarray(image, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("image must be finite [0,1] HxWx3")
    maximum_side = int(config["maximum_side"])
    if maximum_side <= 0:
        raise ValueError("maximum_side must be positive")
    height, width = values.shape[:2]
    if max(height, width) > maximum_side:
        scale = maximum_side / max(height, width)
        values = resize_srgb(
            values,
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
    rows = int(config["spatial_grid_rows"])
    columns = int(config["spatial_grid_columns"])
    if rows <= 0 or columns <= 0 or values.shape[0] < rows or values.shape[1] < columns:
        raise ValueError("invalid spatial descriptor grid")
    wrapper = {"source_descriptor": config}
    descriptors: list[np.ndarray] = []
    if bool(config["include_global_cell"]):
        descriptors.append(source_descriptor(values.reshape(-1, 3), wrapper))
    y_edges = np.linspace(0, values.shape[0], rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, values.shape[1], columns + 1, dtype=np.int64)
    for row in range(rows):
        for column in range(columns):
            cell = values[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            descriptors.append(source_descriptor(cell.reshape(-1, 3), wrapper))
    result = np.concatenate(descriptors)
    if not np.all(np.isfinite(result)):
        raise ValueError("spatial descriptor is non-finite")
    return result


def fit_pca(values: np.ndarray, component_count: int) -> dict[str, np.ndarray]:
    """Fit deterministic PCA with an explicit component-sign convention."""

    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or not np.all(np.isfinite(matrix))
        or component_count <= 0
        or component_count > min(matrix.shape)
    ):
        raise ValueError("invalid PCA input or component count")
    mean = np.mean(matrix, axis=0)
    _, singular_values, right = np.linalg.svd(matrix - mean, full_matrices=False)
    components = right[:component_count].copy()
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            components[index] *= -1.0
    return {
        "mean": mean,
        "components": components,
        "singular_values": singular_values[:component_count],
    }


def encode_pca(model: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    encoded = (matrix - np.asarray(model["mean"])) @ np.asarray(
        model["components"]
    ).T
    if not np.all(np.isfinite(encoded)):
        raise ValueError("PCA encoding is non-finite")
    return encoded


def train_factorized_models(
    after_features: np.ndarray,
    paired_deltas: np.ndarray,
    operator_parameters: np.ndarray,
    *,
    component_count: int,
    student_alpha: float,
    operator_alpha: float,
    cyclic_shift: int,
) -> dict[str, Any]:
    """Fit candidate and matched controls from canonical fit rows only."""

    pca = fit_pca(paired_deltas, component_count)
    latent = encode_pca(pca, paired_deltas)
    return {
        "pca": pca,
        "student": fit_ridge(after_features, latent, alpha=student_alpha),
        "latent_to_operator": fit_ridge(latent, operator_parameters, alpha=operator_alpha),
        "direct": fit_ridge(after_features, operator_parameters, alpha=operator_alpha),
        "permuted_student": fit_ridge(
            after_features,
            np.roll(latent, int(cyclic_shift), axis=0),
            alpha=student_alpha,
        ),
        "global_parameters": np.mean(operator_parameters, axis=0),
    }


def predict_factorized(model: Mapping[str, Any], after_features: np.ndarray) -> dict[str, np.ndarray]:
    candidate_latent = predict_ridge(model["student"], after_features)
    permuted_latent = predict_ridge(model["permuted_student"], after_features)
    return {
        "candidate_latent": candidate_latent,
        "candidate_parameters": predict_ridge(
            model["latent_to_operator"], candidate_latent
        ),
        "direct_parameters": predict_ridge(model["direct"], after_features),
        "permuted_parameters": predict_ridge(
            model["latent_to_operator"], permuted_latent
        ),
    }


__all__ = [
    "encode_pca",
    "fit_pca",
    "fit_ridge",
    "predict_factorized",
    "predict_ridge",
    "spatial_descriptor",
    "train_factorized_models",
]
