"""Explicit source-free global logit-affine preference operator for SPCP2."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass

import numpy as np

EPSILON = 1.0 / 4096.0


@dataclass(frozen=True)
class LogitAffineOperator:
    matrix: np.ndarray
    bias: np.ndarray
    dose: float


def logit_code(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values64, EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values64)
    positive = values64 >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values64[positive]))
    exponential = np.exp(values64[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def sample_indexes(scene_id: str, pixel_count: int, count: int) -> np.ndarray:
    if count > pixel_count:
        raise ValueError("image contains fewer pixels than the frozen sample count")
    prefix = f"{scene_id}|".encode()
    selected = heapq.nsmallest(
        count,
        range(pixel_count),
        key=lambda index: hashlib.sha256(prefix + str(index).encode()).digest(),
    )
    return np.asarray(selected, dtype=np.int64)


def fit_operator(
    sources: list[np.ndarray],
    targets: list[np.ndarray],
    scene_ids: list[str],
    *,
    pixels_per_scene: int,
    ridge_alpha: float,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(sources) == len(targets) == len(scene_ids)) or not sources:
        raise ValueError("fit roles are empty or misaligned")
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for source, target, scene_id in zip(sources, targets, scene_ids, strict=True):
        if source.shape != target.shape or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("fit pair geometry is invalid")
        indexes = sample_indexes(scene_id, source.shape[0] * source.shape[1], pixels_per_scene)
        source_rows.append(logit_code(source.reshape(-1, 3)[indexes]))
        target_rows.append(logit_code(target.reshape(-1, 3)[indexes]))
    x = np.concatenate(source_rows, axis=0)
    y = np.concatenate(target_rows, axis=0)
    if diagonal:
        matrix = np.zeros((3, 3), dtype=np.float64)
        bias = np.zeros(3, dtype=np.float64)
        for channel in range(3):
            design = np.column_stack((x[:, channel], np.ones(x.shape[0])))
            gram = design.T @ design + ridge_alpha * np.eye(2, dtype=np.float64)
            coefficients = np.linalg.solve(gram, design.T @ y[:, channel])
            matrix[channel, channel], bias[channel] = coefficients
        return matrix, bias
    design = np.column_stack((x, np.ones(x.shape[0])))
    gram = design.T @ design + ridge_alpha * np.eye(4, dtype=np.float64)
    coefficients = np.linalg.solve(gram, design.T @ y)
    return coefficients[:3, :].T.copy(), coefficients[3, :].copy()


def fit_operator_rows(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    *,
    ridge_alpha: float,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = logit_code(np.asarray(source_rows, dtype=np.float64))
    y = logit_code(np.asarray(target_rows, dtype=np.float64))
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("sample rows must be aligned Nx3 arrays")
    if diagonal:
        matrix = np.zeros((3, 3), dtype=np.float64)
        bias = np.zeros(3, dtype=np.float64)
        for channel in range(3):
            design = np.column_stack((x[:, channel], np.ones(x.shape[0])))
            gram = design.T @ design + ridge_alpha * np.eye(2, dtype=np.float64)
            coefficients = np.linalg.solve(gram, design.T @ y[:, channel])
            matrix[channel, channel], bias[channel] = coefficients
        return matrix, bias
    design = np.column_stack((x, np.ones(x.shape[0])))
    gram = design.T @ design + ridge_alpha * np.eye(4, dtype=np.float64)
    coefficients = np.linalg.solve(gram, design.T @ y)
    return coefficients[:3, :].T.copy(), coefficients[3, :].copy()


def dose_operator(matrix: np.ndarray, bias: np.ndarray, dose: float) -> LogitAffineOperator:
    identity = np.eye(3, dtype=np.float64)
    return LogitAffineOperator(
        matrix=identity + dose * (np.asarray(matrix, dtype=np.float64) - identity),
        bias=dose * np.asarray(bias, dtype=np.float64),
        dose=float(dose),
    )


def apply_operator(source: np.ndarray, operator: LogitAffineOperator) -> np.ndarray:
    source64 = np.asarray(source, dtype=np.float64)
    transformed = logit_code(source64) @ operator.matrix.T + operator.bias
    return sigmoid(transformed).astype(np.float32)


def matrix_diagnostics(operator: LogitAffineOperator) -> dict[str, float]:
    singular = np.linalg.svd(operator.matrix, compute_uv=False)
    return {
        "determinant": float(np.linalg.det(operator.matrix)),
        "condition_number": float(singular[0] / singular[-1]),
        "minimum_singular_value": float(singular[-1]),
    }


def srgb_code_to_oklab(source: np.ndarray) -> np.ndarray:
    code = np.asarray(source, dtype=np.float64)
    linear = np.where(code <= 0.04045, code / 12.92, ((code + 0.055) / 1.055) ** 2.4)
    lms = linear @ np.asarray(
        (
            (0.4122214708, 0.2119034982, 0.0883024619),
            (0.5363325363, 0.6806995451, 0.2817188376),
            (0.0514459929, 0.1073969566, 0.6299787005),
        ),
        dtype=np.float64,
    )
    lms_root = np.cbrt(lms)
    return lms_root @ np.asarray(
        (
            (0.2104542553, 1.9779984951, 0.0259040371),
            (0.7936177850, -2.4285922050, 0.7827717662),
            (-0.0040720468, 0.4505937099, -0.8086757660),
        ),
        dtype=np.float64,
    )


def mean_oklab_error(candidate: np.ndarray, target: np.ndarray) -> float:
    delta = srgb_code_to_oklab(candidate) - srgb_code_to_oklab(target)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))


def gradient_p999_ratio(source: np.ndarray, output: np.ndarray) -> float:
    def magnitudes(image: np.ndarray) -> np.ndarray:
        luma = np.asarray(image, dtype=np.float64) @ np.asarray((0.2126, 0.7152, 0.0722))
        dx = np.diff(luma, axis=1).ravel()
        dy = np.diff(luma, axis=0).ravel()
        return np.concatenate((np.abs(dx), np.abs(dy)))

    denominator = float(np.quantile(magnitudes(source), 0.999))
    numerator = float(np.quantile(magnitudes(output), 0.999))
    return numerator / max(denominator, 1.0e-12)


def new_exact_boundary_fraction(source: np.ndarray, output: np.ndarray) -> float:
    source_boundary = (source <= 0.0) | (source >= 1.0)
    output_boundary = (output <= 0.0) | (output >= 1.0)
    return float(np.mean(output_boundary & ~source_boundary))
