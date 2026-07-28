"""Deterministic paired fitting for the explicit positive-film response operator.

This module estimates only the observable endpoint-normalized mapping.  The
sigmoid amplitude is fixed to one because it is gauge-equivalent to the scan
matrix after per-output endpoint normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from .positive_film import PositiveFilmResponseOperator


PositiveFilmFitModel = Literal["two_matrix", "one_matrix"]


@dataclass(frozen=True)
class PositiveFilmFitResult:
    """One deterministic paired fit and its development-set evidence."""

    operator: PositiveFilmResponseOperator
    model: PositiveFilmFitModel
    development_rgb_rmse: float
    development_maximum_absolute_error: float
    optimization_cost: float
    optimality: float
    function_evaluations: int
    restart_index: int
    converged: bool


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def row_stochastic_identity_mixture(
    free_logits: np.ndarray,
    *,
    identity_mixture: float,
) -> np.ndarray:
    """Decode six off-diagonal logits into a safe row-stochastic 3x3 matrix.

    The diagonal logit is fixed at zero, removing the softmax row-offset gauge.
    At ``identity_mixture <= 0.25`` every possible decoded matrix has
    determinant at least 0.25, so the J0 determinant contract is structural
    rather than an optimizer penalty.
    """

    values = np.asarray(free_logits, dtype=np.float64)
    if (
        values.shape != (6,)
        or not np.all(np.isfinite(values))
        or not np.isfinite(identity_mixture)
        or identity_mixture <= 0.0
        or identity_mixture > 0.25
    ):
        raise ValueError(
            "free_logits must be six finite values and identity_mixture in (0, .25]"
        )
    logits = np.zeros((3, 3), dtype=np.float64)
    cursor = 0
    for row in range(3):
        for column in range(3):
            if column != row:
                logits[row, column] = values[cursor]
                cursor += 1
    probabilities = _softmax(logits)
    return (1.0 - identity_mixture) * np.eye(3) + identity_mixture * probabilities


def _parameter_count(model: PositiveFilmFitModel) -> int:
    if model == "two_matrix":
        return 18
    if model == "one_matrix":
        return 12
    raise ValueError("model must be 'two_matrix' or 'one_matrix'")


def _decode_parameters(
    parameters: np.ndarray,
    *,
    model: PositiveFilmFitModel,
    identity_mixture: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (_parameter_count(model),) or not np.all(np.isfinite(values)):
        raise ValueError("parameters have the wrong shape or contain non-finite values")
    capture = row_stochastic_identity_mixture(
        values[:6], identity_mixture=identity_mixture
    )
    if model == "two_matrix":
        scan = row_stochastic_identity_mixture(
            values[6:12], identity_mixture=identity_mixture
        )
        curve_offset = 12
    else:
        scan = np.eye(3, dtype=np.float64)
        curve_offset = 6
    midpoints = values[curve_offset : curve_offset + 3]
    slopes = values[curve_offset + 3 : curve_offset + 6]
    return capture, scan, midpoints, slopes


def _apply_decoded(
    linear_rgb: np.ndarray,
    *,
    capture_matrix: np.ndarray,
    scan_matrix: np.ndarray,
    response_midpoints: np.ndarray,
    response_slopes: np.ndarray,
    exposure_floor: float,
) -> np.ndarray:
    layer_exposure = linear_rgb @ capture_matrix.T
    log_exposure = np.log2(layer_exposure + exposure_floor)
    response = 1.0 / (
        1.0
        + np.exp(
            -response_slopes[None, :]
            * (log_exposure - response_midpoints[None, :])
        )
    )
    raw = response @ scan_matrix.T

    endpoints = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    endpoint_exposure = endpoints @ capture_matrix.T
    endpoint_log = np.log2(endpoint_exposure + exposure_floor)
    endpoint_response = 1.0 / (
        1.0
        + np.exp(
            -response_slopes[None, :]
            * (endpoint_log - response_midpoints[None, :])
        )
    )
    endpoint_raw = endpoint_response @ scan_matrix.T
    return (raw - endpoint_raw[0]) / (endpoint_raw[1] - endpoint_raw[0])


def _validate_pairs(
    source_rgb: np.ndarray, target_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_rgb, dtype=np.float64)
    target = np.asarray(target_rgb, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or target.shape != source.shape
        or source.shape[0] < 12
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(target < 0.0)
        or np.any(target > 1.0)
    ):
        raise ValueError(
            "paired source/target must be finite [0,1] arrays of shape (N,3), N>=12"
        )
    return source, target


def _initial_parameters(
    *,
    model: PositiveFilmFitModel,
    restart_index: int,
    seed: int,
) -> np.ndarray:
    count = _parameter_count(model)
    values = np.zeros(count, dtype=np.float64)
    matrix_parameter_count = 12 if model == "two_matrix" else 6
    values[:matrix_parameter_count] = np.log(1.0 / 8.0)
    curve_offset = matrix_parameter_count
    values[curve_offset : curve_offset + 3] = -3.0
    values[curve_offset + 3 : curve_offset + 6] = 1.0
    if restart_index:
        rng = np.random.default_rng(seed + restart_index)
        values[:matrix_parameter_count] += rng.normal(
            0.0, 0.35, matrix_parameter_count
        )
        values[curve_offset : curve_offset + 3] += rng.normal(0.0, 0.35, 3)
        values[curve_offset + 3 : curve_offset + 6] += rng.normal(0.0, 0.15, 3)
    return values


def fit_positive_film_response_operator(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    model: PositiveFilmFitModel,
    identity_mixture: float = 0.25,
    restart_count: int = 3,
    maximum_function_evaluations: int = 2500,
    function_tolerance: float = 1e-11,
    parameter_tolerance: float = 1e-11,
    gradient_tolerance: float = 1e-11,
    seed: int = 20250728,
) -> PositiveFilmFitResult:
    """Fit the J0 operator family to controlled paired RGB observations."""

    source, target = _validate_pairs(source_rgb, target_rgb)
    if (
        model not in ("two_matrix", "one_matrix")
        or not isinstance(restart_count, int)
        or restart_count < 1
        or not isinstance(maximum_function_evaluations, int)
        or maximum_function_evaluations < 1
        or any(
            not np.isfinite(value) or value <= 0.0
            for value in (function_tolerance, parameter_tolerance, gradient_tolerance)
        )
    ):
        raise ValueError("invalid paired-fit controls")
    # Validate the structural determinant contract before optimization.
    row_stochastic_identity_mixture(
        np.zeros(6, dtype=np.float64), identity_mixture=identity_mixture
    )

    matrix_parameter_count = 12 if model == "two_matrix" else 6
    lower = np.concatenate(
        (
            np.full(matrix_parameter_count, -8.0),
            np.full(3, -12.0),
            np.full(3, 0.2),
        )
    )
    upper = np.concatenate(
        (
            np.full(matrix_parameter_count, 4.0),
            np.full(3, 2.0),
            np.full(3, 4.0),
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        capture, scan, midpoints, slopes = _decode_parameters(
            parameters, model=model, identity_mixture=identity_mixture
        )
        prediction = _apply_decoded(
            source,
            capture_matrix=capture,
            scan_matrix=scan,
            response_midpoints=midpoints,
            response_slopes=slopes,
            exposure_floor=2.0**-16,
        )
        return (prediction - target).reshape(-1)

    best_result = None
    best_restart = -1
    for restart_index in range(restart_count):
        initial = np.clip(
            _initial_parameters(
                model=model, restart_index=restart_index, seed=seed
            ),
            lower + 1e-9,
            upper - 1e-9,
        )
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            x_scale="jac",
        )
        if best_result is None or result.cost < best_result.cost:
            best_result = result
            best_restart = restart_index
    assert best_result is not None

    capture, scan, midpoints, slopes = _decode_parameters(
        best_result.x, model=model, identity_mixture=identity_mixture
    )
    operator = PositiveFilmResponseOperator(
        capture_matrix=capture,
        response_midpoints=midpoints,
        response_slopes=slopes,
        maximum_responses=np.ones(3, dtype=np.float64),
        scan_matrix=scan,
    )
    error = operator.apply(source) - target
    return PositiveFilmFitResult(
        operator=operator,
        model=model,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        development_maximum_absolute_error=float(np.max(np.abs(error))),
        optimization_cost=float(best_result.cost),
        optimality=float(best_result.optimality),
        function_evaluations=int(best_result.nfev),
        restart_index=best_restart,
        converged=bool(best_result.success),
    )


__all__ = [
    "PositiveFilmFitModel",
    "PositiveFilmFitResult",
    "fit_positive_film_response_operator",
    "row_stochastic_identity_mixture",
]
