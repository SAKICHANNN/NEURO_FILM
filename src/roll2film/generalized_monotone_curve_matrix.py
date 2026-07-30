"""Variable-resolution positive monotone curves followed by a positive matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.roll2film.positive_film_fitting import (
    PositiveFilmFitLoss,
    row_stochastic_identity_mixture,
)


SCHEMA = "roll2film.generalized_monotone_curve_positive_matrix.v1"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _rgb(values: np.ndarray) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float64)
    if (
        rgb.ndim < 1
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ValueError("RGB must be finite [0, 1] data")
    return rgb


@dataclass(frozen=True)
class GeneralizedMonotoneCurveMatrixOperator:
    curve_segment_weights: np.ndarray
    matrix: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.curve_segment_weights, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if (
            weights.ndim != 2
            or weights.shape[0] != 3
            or weights.shape[1] < 3
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12)
        ):
            raise ValueError("curve weights must be positive 3xS probability rows")
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-12)
            or np.linalg.det(matrix) <= 0.0
        ):
            raise ValueError("matrix must be positive row-stochastic and oriented")
        weights = weights.copy()
        matrix = matrix.copy()
        weights.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "curve_segment_weights", weights)
        object.__setattr__(self, "matrix", matrix)

    @property
    def segment_count(self) -> int:
        return int(self.curve_segment_weights.shape[1])

    def _curves(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _rgb(rgb)
        flat = values.reshape(-1, 3)
        segments = self.segment_count
        scaled = flat * float(segments)
        indices = np.minimum(np.floor(scaled).astype(np.int64), segments - 1)
        fractions = scaled - indices
        fractions[flat == 1.0] = 1.0
        cumulative = np.concatenate(
            (
                np.zeros((3, 1), dtype=np.float64),
                np.cumsum(self.curve_segment_weights, axis=1),
            ),
            axis=1,
        )
        channels = np.arange(3)[None, :]
        curved = (
            cumulative[channels, indices]
            + fractions * self.curve_segment_weights[channels, indices]
        )
        return curved.reshape(values.shape), indices

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        curved, _ = self._curves(rgb)
        output = curved @ self.matrix.T
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise RuntimeError("generalized monotone operator escaped cube")
        return np.clip(output, 0.0, 1.0)

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        values = _rgb(rgb)
        _, indices = self._curves(values)
        channels = np.arange(3)[None, :]
        slopes = (
            float(self.segment_count)
            * self.curve_segment_weights[channels, indices]
        )
        return (
            float(np.linalg.det(self.matrix)) * np.prod(slopes, axis=1)
        ).reshape(values.shape[:-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "segment_count": self.segment_count,
            "curve_segment_weights": self.curve_segment_weights.tolist(),
            "matrix": self.matrix.tolist(),
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "GeneralizedMonotoneCurveMatrixOperator":
        if (
            payload.get("schema") != SCHEMA
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("unsupported generalized monotone schema")
        operator = cls(
            curve_segment_weights=np.asarray(
                payload["curve_segment_weights"], dtype=np.float64
            ),
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
        )
        if operator.segment_count != int(payload["segment_count"]):
            raise ValueError("segment count drift")
        return operator


@dataclass(frozen=True)
class GeneralizedMonotoneFitResult:
    operator: (
        GeneralizedMonotoneCurveMatrixOperator
        | PositiveMatrixCurveMatrixOperator
    )
    development_rgb_rmse: float
    function_evaluations: int
    restart_index: int
    converged: bool


@dataclass(frozen=True)
class PositiveMatrixCurveMatrixOperator:
    """Positive matrix, monotone curves, then a second positive matrix."""

    input_matrix: np.ndarray
    curve_segment_weights: np.ndarray
    output_matrix: np.ndarray

    def __post_init__(self) -> None:
        helper = GeneralizedMonotoneCurveMatrixOperator(
            self.curve_segment_weights, self.output_matrix
        )
        input_matrix = np.asarray(self.input_matrix, dtype=np.float64)
        if (
            input_matrix.shape != (3, 3)
            or not np.all(np.isfinite(input_matrix))
            or np.any(input_matrix < 0.0)
            or not np.allclose(
                np.sum(input_matrix, axis=1), 1.0, atol=1e-12
            )
            or np.linalg.det(input_matrix) <= 0.0
        ):
            raise ValueError("input matrix must be positive and oriented")
        input_matrix = input_matrix.copy()
        input_matrix.setflags(write=False)
        object.__setattr__(self, "input_matrix", input_matrix)
        object.__setattr__(
            self, "curve_segment_weights", helper.curve_segment_weights
        )
        object.__setattr__(self, "output_matrix", helper.matrix)

    @property
    def segment_count(self) -> int:
        return int(self.curve_segment_weights.shape[1])

    def _intermediate(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _rgb(rgb)
        mixed = values @ self.input_matrix.T
        helper = GeneralizedMonotoneCurveMatrixOperator(
            self.curve_segment_weights, np.eye(3)
        )
        curved, indices = helper._curves(mixed)
        return curved, indices

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        curved, _ = self._intermediate(rgb)
        output = curved @ self.output_matrix.T
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise RuntimeError("matrix-curve-matrix operator escaped cube")
        return np.clip(output, 0.0, 1.0)

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        values = _rgb(rgb)
        _, indices = self._intermediate(values)
        channels = np.arange(3)[None, :]
        slopes = (
            float(self.segment_count)
            * self.curve_segment_weights[channels, indices]
        )
        return (
            float(np.linalg.det(self.input_matrix))
            * float(np.linalg.det(self.output_matrix))
            * np.prod(slopes, axis=1)
        ).reshape(values.shape[:-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "roll2film.positive_matrix_curve_matrix.v1",
            "segment_count": self.segment_count,
            "input_matrix": self.input_matrix.tolist(),
            "curve_segment_weights": self.curve_segment_weights.tolist(),
            "output_matrix": self.output_matrix.tolist(),
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "PositiveMatrixCurveMatrixOperator":
        if (
            payload.get("schema")
            != "roll2film.positive_matrix_curve_matrix.v1"
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("unsupported matrix-curve-matrix schema")
        operator = cls(
            input_matrix=np.asarray(payload["input_matrix"], dtype=np.float64),
            curve_segment_weights=np.asarray(
                payload["curve_segment_weights"], dtype=np.float64
            ),
            output_matrix=np.asarray(
                payload["output_matrix"], dtype=np.float64
            ),
        )
        if operator.segment_count != int(payload["segment_count"]):
            raise ValueError("segment count drift")
        return operator


def fit_generalized_monotone_curve_matrix(
    source: np.ndarray,
    target: np.ndarray,
    *,
    segment_count: int,
    curve_learned_mixture: float,
    matrix_identity_mixture: float,
    free_logit_bounds: tuple[float, float],
    restart_count: int,
    maximum_function_evaluations: int,
    function_tolerance: float,
    parameter_tolerance: float,
    gradient_tolerance: float,
    loss: PositiveFilmFitLoss,
    loss_scale: float,
    seed: int,
) -> GeneralizedMonotoneFitResult:
    source_values = _rgb(source)
    target_values = _rgb(target)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or len(source_values) < 3 * segment_count
        or segment_count < 3
        or not 0.0 < curve_learned_mixture < 1.0
    ):
        raise ValueError("invalid generalized monotone fit data")
    lower, upper = map(float, free_logit_bounds)
    curve_parameters = 3 * (segment_count - 1)
    parameter_count = curve_parameters + 6

    def operator(parameters: np.ndarray) -> GeneralizedMonotoneCurveMatrixOperator:
        logits = np.zeros((3, segment_count), dtype=np.float64)
        logits[:, 1:] = parameters[:curve_parameters].reshape(
            3, segment_count - 1
        )
        probabilities = _softmax(logits)
        weights = (
            (1.0 - curve_learned_mixture) / float(segment_count)
            + curve_learned_mixture * probabilities
        )
        matrix = row_stochastic_identity_mixture(
            parameters[curve_parameters:],
            identity_mixture=matrix_identity_mixture,
        )
        return GeneralizedMonotoneCurveMatrixOperator(weights, matrix)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (operator(parameters).apply(source_values) - target_values).reshape(
            -1
        )

    best = None
    best_operator = None
    best_restart = -1
    for restart in range(restart_count):
        initial = np.zeros(parameter_count, dtype=np.float64)
        initial[curve_parameters:] = np.log(1.0 / 8.0)
        if restart:
            rng = np.random.default_rng(seed + restart)
            initial += rng.normal(0.0, 0.25, parameter_count)
        initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            residual,
            initial,
            bounds=(
                np.full(parameter_count, lower),
                np.full(parameter_count, upper),
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
        candidate = operator(result.x)
        if best is None or result.cost < best.cost:
            best = result
            best_operator = candidate
            best_restart = restart
    if best is None or best_operator is None:
        raise RuntimeError("generalized monotone fit produced no result")
    error = best_operator.apply(source_values) - target_values
    return GeneralizedMonotoneFitResult(
        operator=best_operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        function_evaluations=int(best.nfev),
        restart_index=best_restart,
        converged=bool(best.success),
    )


def fit_positive_matrix_curve_matrix(
    source: np.ndarray,
    target: np.ndarray,
    *,
    segment_count: int,
    curve_learned_mixture: float,
    input_matrix_identity_mixture: float,
    output_matrix_identity_mixture: float,
    free_logit_bounds: tuple[float, float],
    restart_count: int,
    maximum_function_evaluations: int,
    function_tolerance: float,
    parameter_tolerance: float,
    gradient_tolerance: float,
    loss: PositiveFilmFitLoss,
    loss_scale: float,
    seed: int,
) -> GeneralizedMonotoneFitResult:
    source_values = _rgb(source)
    target_values = _rgb(target)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or len(source_values) < 3 * segment_count
        or segment_count < 3
        or not 0.0 < curve_learned_mixture < 1.0
        or not 0.0 <= input_matrix_identity_mixture <= 1.0
        or not 0.0 <= output_matrix_identity_mixture <= 1.0
        or restart_count < 1
        or maximum_function_evaluations < 1
    ):
        raise ValueError("invalid matrix-curve-matrix fit data")
    lower, upper = map(float, free_logit_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("invalid matrix-curve-matrix logit bounds")
    curve_parameters = 3 * (segment_count - 1)
    parameter_count = curve_parameters + 12

    def operator(parameters: np.ndarray) -> PositiveMatrixCurveMatrixOperator:
        logits = np.zeros((3, segment_count), dtype=np.float64)
        logits[:, 1:] = parameters[:curve_parameters].reshape(
            3, segment_count - 1
        )
        probabilities = _softmax(logits)
        weights = (
            (1.0 - curve_learned_mixture) / float(segment_count)
            + curve_learned_mixture * probabilities
        )
        return PositiveMatrixCurveMatrixOperator(
            input_matrix=row_stochastic_identity_mixture(
                parameters[curve_parameters : curve_parameters + 6],
                identity_mixture=input_matrix_identity_mixture,
            ),
            curve_segment_weights=weights,
            output_matrix=row_stochastic_identity_mixture(
                parameters[curve_parameters + 6 :],
                identity_mixture=output_matrix_identity_mixture,
            ),
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (operator(parameters).apply(source_values) - target_values).reshape(
            -1
        )

    best = None
    best_operator = None
    best_restart = -1
    for restart in range(restart_count):
        initial = np.zeros(parameter_count, dtype=np.float64)
        initial[curve_parameters:] = np.log(1.0 / 8.0)
        if restart:
            rng = np.random.default_rng(seed + restart)
            initial += rng.normal(0.0, 0.25, parameter_count)
        initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            residual,
            initial,
            bounds=(
                np.full(parameter_count, lower),
                np.full(parameter_count, upper),
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
        candidate = operator(result.x)
        if best is None or result.cost < best.cost:
            best = result
            best_operator = candidate
            best_restart = restart
    if best is None or best_operator is None:
        raise RuntimeError("matrix-curve-matrix fit produced no result")
    error = best_operator.apply(source_values) - target_values
    return GeneralizedMonotoneFitResult(
        operator=best_operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        function_evaluations=int(best.nfev),
        restart_index=best_restart,
        converged=bool(best.success),
    )


__all__ = [
    "GeneralizedMonotoneCurveMatrixOperator",
    "GeneralizedMonotoneFitResult",
    "PositiveMatrixCurveMatrixOperator",
    "fit_generalized_monotone_curve_matrix",
    "fit_positive_matrix_curve_matrix",
]
