"""Bounded monotone channel curves followed by one positive RGB matrix.

This is a clean-room explicit baseline inspired only by the equation order
described in FILM2PAINT.  It does not reproduce that work's unavailable code,
data, knots, fitted parameters, or Adobe-RGB processing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .positive_film_fitting import (
    PositiveFilmFitLoss,
    row_stochastic_identity_mixture,
)


_SCHEMA = "roll2film.monotone_curve_positive_matrix.v1"
_SEGMENT_COUNT = 3


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _validate_rgb(values: np.ndarray, *, name: str) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float64)
    if (
        rgb.ndim < 1
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ValueError(f"{name} must be a finite [0,1] array with RGB last")
    return rgb


@dataclass(frozen=True)
class MonotoneCurvePositiveMatrixOperator:
    """Endpoint-fixed piecewise-linear curves followed by a positive matrix."""

    curve_segment_weights: np.ndarray
    matrix: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.curve_segment_weights, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if (
            weights.shape != (3, _SEGMENT_COUNT)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
        ):
            raise ValueError(
                "curve_segment_weights must be positive finite 3x3 rows summing to one"
            )
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-12)
            or float(np.linalg.det(matrix)) <= 0.0
        ):
            raise ValueError(
                "matrix must be finite, nonnegative, row-stochastic and positive determinant"
            )
        object.__setattr__(self, "curve_segment_weights", weights.copy())
        object.__setattr__(self, "matrix", matrix.copy())

    @staticmethod
    def _curve_apply(
        rgb: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        flat = rgb.reshape(-1, 3)
        scaled = flat * float(_SEGMENT_COUNT)
        indices = np.minimum(
            np.floor(scaled).astype(np.int64), _SEGMENT_COUNT - 1
        )
        fractions = scaled - indices
        fractions[flat == 1.0] = 1.0
        cumulative = np.concatenate(
            (np.zeros((3, 1), dtype=np.float64), np.cumsum(weights, axis=1)),
            axis=1,
        )
        channels = np.arange(3, dtype=np.int64)[None, :]
        bases = cumulative[channels, indices]
        increments = weights[channels, indices]
        curved = bases + fractions * increments
        return curved.reshape(rgb.shape), indices

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        rgb = _validate_rgb(linear_rgb, name="linear_rgb")
        curved, _ = self._curve_apply(rgb, self.curve_segment_weights)
        output = curved @ self.matrix.T
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise ValueError("bounded operator produced an invalid RGB value")
        return np.clip(output, 0.0, 1.0)

    def inverse(self, output_rgb: np.ndarray) -> np.ndarray:
        output = _validate_rgb(output_rgb, name="output_rgb")
        flat = output.reshape(-1, 3)
        curved = flat @ np.linalg.inv(self.matrix).T
        if (
            not np.all(np.isfinite(curved))
            or np.any(curved < -1e-10)
            or np.any(curved > 1.0 + 1e-10)
        ):
            raise ValueError("output_rgb is outside the operator image")
        curved = np.clip(curved, 0.0, 1.0)
        source = np.empty_like(curved)
        cumulative = np.concatenate(
            (
                np.zeros((3, 1), dtype=np.float64),
                np.cumsum(self.curve_segment_weights, axis=1),
            ),
            axis=1,
        )
        for channel in range(3):
            values = curved[:, channel]
            indices = np.searchsorted(
                cumulative[channel, 1:], values, side="left"
            )
            indices = np.minimum(indices, _SEGMENT_COUNT - 1)
            at_white = values == 1.0
            indices[at_white] = _SEGMENT_COUNT - 1
            fractions = (
                values - cumulative[channel, indices]
            ) / self.curve_segment_weights[channel, indices]
            fractions[at_white] = 1.0
            source[:, channel] = (
                indices.astype(np.float64) + fractions
            ) / float(_SEGMENT_COUNT)
        return source.reshape(output.shape)

    def jacobian_determinants(self, linear_rgb: np.ndarray) -> np.ndarray:
        rgb = _validate_rgb(linear_rgb, name="linear_rgb")
        _, indices = self._curve_apply(rgb, self.curve_segment_weights)
        channels = np.arange(3, dtype=np.int64)[None, :]
        slopes = (
            float(_SEGMENT_COUNT)
            * self.curve_segment_weights[channels, indices]
        )
        determinants = float(np.linalg.det(self.matrix)) * np.prod(
            slopes, axis=1
        )
        return determinants.reshape(rgb.shape[:-1])

    def to_dict(self) -> dict:
        return {
            "schema": _SCHEMA,
            "curve_x_knots": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            "curve_segment_weights": self.curve_segment_weights.tolist(),
            "matrix": self.matrix.tolist(),
            "operator_order": ["monotone_channel_curves", "positive_matrix"],
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MonotoneCurvePositiveMatrixOperator":
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _SCHEMA
            or payload.get("curve_x_knots")
            != [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
            or payload.get("operator_order")
            != ["monotone_channel_curves", "positive_matrix"]
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("invalid monotone-curve matrix payload")
        return cls(
            curve_segment_weights=np.asarray(
                payload.get("curve_segment_weights"), dtype=np.float64
            ),
            matrix=np.asarray(payload.get("matrix"), dtype=np.float64),
        )


@dataclass(frozen=True)
class MonotoneCurveMatrixFitResult:
    operator: MonotoneCurvePositiveMatrixOperator
    development_rgb_rmse: float
    development_maximum_absolute_error: float
    optimization_cost: float
    optimality: float
    function_evaluations: int
    restart_index: int
    converged: bool
    loss: PositiveFilmFitLoss
    loss_scale: float


def _decode_parameters(
    parameters: np.ndarray,
    *,
    curve_identity_mixture: float,
    matrix_identity_mixture: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain twelve finite values")
    logits = np.zeros((3, 3), dtype=np.float64)
    logits[:, 1:] = values[:6].reshape(3, 2)
    probabilities = _softmax(logits)
    weights = (
        (1.0 - curve_identity_mixture) / float(_SEGMENT_COUNT)
        + curve_identity_mixture * probabilities
    )
    matrix = row_stochastic_identity_mixture(
        values[6:], identity_mixture=matrix_identity_mixture
    )
    return weights, matrix


def _validate_pairs(
    source_rgb: np.ndarray, target_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = _validate_rgb(source_rgb, name="source_rgb")
    target = _validate_rgb(target_rgb, name="target_rgb")
    if source.ndim != 2 or target.shape != source.shape or source.shape[0] < 12:
        raise ValueError("paired RGB must have shape (N,3), N>=12")
    return source, target


def fit_monotone_curve_positive_matrix(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    curve_identity_mixture: float = 0.75,
    matrix_identity_mixture: float = 0.25,
    free_logit_bounds: tuple[float, float] = (-8.0, 4.0),
    restart_count: int = 5,
    maximum_function_evaluations: int = 3000,
    function_tolerance: float = 1e-11,
    parameter_tolerance: float = 1e-11,
    gradient_tolerance: float = 1e-11,
    loss: PositiveFilmFitLoss = "linear",
    loss_scale: float = 0.01,
    seed: int = 20250729,
) -> MonotoneCurveMatrixFitResult:
    """Fit the twelve-parameter bounded curve-then-matrix family."""

    source, target = _validate_pairs(source_rgb, target_rgb)
    lower, upper = (float(v) for v in free_logit_bounds)
    if (
        not np.isfinite(curve_identity_mixture)
        or curve_identity_mixture <= 0.0
        or curve_identity_mixture >= 1.0
        or not np.isfinite(matrix_identity_mixture)
        or matrix_identity_mixture <= 0.0
        or matrix_identity_mixture > 0.25
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower >= upper
        or not isinstance(restart_count, int)
        or restart_count < 1
        or not isinstance(maximum_function_evaluations, int)
        or maximum_function_evaluations < 1
        or loss not in ("linear", "soft_l1", "huber", "cauchy", "arctan")
        or not np.isfinite(loss_scale)
        or loss_scale <= 0.0
        or any(
            not np.isfinite(value) or value <= 0.0
            for value in (
                function_tolerance,
                parameter_tolerance,
                gradient_tolerance,
            )
        )
    ):
        raise ValueError("invalid monotone-curve matrix fit controls")

    def make_operator(parameters: np.ndarray) -> MonotoneCurvePositiveMatrixOperator:
        weights, matrix = _decode_parameters(
            parameters,
            curve_identity_mixture=curve_identity_mixture,
            matrix_identity_mixture=matrix_identity_mixture,
        )
        return MonotoneCurvePositiveMatrixOperator(weights, matrix)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (make_operator(parameters).apply(source) - target).reshape(-1)

    best_result = None
    best_operator = None
    best_restart = -1
    for restart_index in range(restart_count):
        initial = np.zeros(12, dtype=np.float64)
        initial[6:] = np.log(1.0 / 8.0)
        if restart_index:
            rng = np.random.default_rng(seed + restart_index)
            initial[:6] += rng.normal(0.0, 0.35, 6)
            initial[6:] += rng.normal(0.0, 0.35, 6)
        initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            residual,
            initial,
            bounds=(
                np.full(12, lower, dtype=np.float64),
                np.full(12, upper, dtype=np.float64),
            ),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            x_scale="jac",
            loss=loss,
            f_scale=loss_scale,
        )
        operator = make_operator(result.x)
        if best_result is None or result.cost < best_result.cost:
            best_result = result
            best_operator = operator
            best_restart = restart_index
    if best_result is None or best_operator is None:
        raise RuntimeError("monotone-curve matrix fit produced no result")
    error = best_operator.apply(source) - target
    return MonotoneCurveMatrixFitResult(
        operator=best_operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        development_maximum_absolute_error=float(np.max(np.abs(error))),
        optimization_cost=float(best_result.cost),
        optimality=float(best_result.optimality),
        function_evaluations=int(best_result.nfev),
        restart_index=best_restart,
        converged=bool(best_result.success),
        loss=loss,
        loss_scale=float(loss_scale),
    )


__all__ = [
    "MonotoneCurveMatrixFitResult",
    "MonotoneCurvePositiveMatrixOperator",
    "fit_monotone_curve_positive_matrix",
]
