"""Structurally bounded two-matrix film equation with a derivative floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .positive_film_fitting import (
    PositiveFilmFitLoss,
    row_stochastic_identity_mixture,
)


SCHEMA = "roll2film.identity_residual_sigmoid.v1"


def _validate_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-12)
        or float(np.linalg.det(matrix)) <= 0.0
    ):
        raise ValueError(f"{name} must be positive-determinant row-stochastic 3x3")
    return matrix.copy()


def _validate_rgb(value: np.ndarray, *, name: str) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)
    if (
        rgb.ndim < 1
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ValueError(f"{name} must be finite [0,1] RGB-last data")
    return rgb


@dataclass(frozen=True)
class IdentityResidualSigmoidOperator:
    capture_matrix: np.ndarray
    response_midpoints: np.ndarray
    response_slopes: np.ndarray
    scan_matrix: np.ndarray
    nonlinear_strength: float = 0.75
    exposure_floor: float = 2.0**-16

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capture_matrix",
            _validate_matrix(self.capture_matrix, name="capture_matrix"),
        )
        object.__setattr__(
            self,
            "scan_matrix",
            _validate_matrix(self.scan_matrix, name="scan_matrix"),
        )
        midpoints = np.asarray(self.response_midpoints, dtype=np.float64)
        slopes = np.asarray(self.response_slopes, dtype=np.float64)
        if (
            midpoints.shape != (3,)
            or slopes.shape != (3,)
            or not np.all(np.isfinite(midpoints))
            or not np.all(np.isfinite(slopes))
            or np.any(midpoints < -12.0)
            or np.any(midpoints > 2.0)
            or np.any(slopes < 0.2)
            or np.any(slopes > 4.0)
        ):
            raise ValueError("midpoints/slopes violate the v1 parameter box")
        object.__setattr__(self, "response_midpoints", midpoints.copy())
        object.__setattr__(self, "response_slopes", slopes.copy())
        if (
            not np.isfinite(self.nonlinear_strength)
            or self.nonlinear_strength <= 0.0
            or self.nonlinear_strength >= 1.0
        ):
            raise ValueError("nonlinear_strength must lie strictly in (0,1)")
        if self.exposure_floor != 2.0**-16:
            raise ValueError("v1 exposure_floor must equal 2^-16")

    def _sigmoid(self, exposure: np.ndarray) -> np.ndarray:
        log_exposure = np.log2(exposure + self.exposure_floor)
        argument = self.response_slopes * (
            log_exposure - self.response_midpoints
        )
        return 1.0 / (1.0 + np.exp(-argument))

    def _normalized_sigmoid(self, exposure: np.ndarray) -> np.ndarray:
        black = self._sigmoid(np.zeros(3, dtype=np.float64))
        white = self._sigmoid(np.ones(3, dtype=np.float64))
        return (self._sigmoid(exposure) - black) / (white - black)

    def apply(self, input_rgb: np.ndarray) -> np.ndarray:
        rgb = _validate_rgb(input_rgb, name="input_rgb")
        layer = rgb @ self.capture_matrix.T
        nonlinear = self._normalized_sigmoid(layer)
        mixed = (
            (1.0 - self.nonlinear_strength) * layer
            + self.nonlinear_strength * nonlinear
        )
        output = mixed @ self.scan_matrix.T
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise ValueError("identity-residual sigmoid escaped the RGB cube")
        return np.clip(output, 0.0, 1.0)

    def jacobian_determinants(self, input_rgb: np.ndarray) -> np.ndarray:
        rgb = _validate_rgb(input_rgb, name="input_rgb")
        flat = rgb.reshape(-1, 3)
        layer = flat @ self.capture_matrix.T
        sigmoid = self._sigmoid(layer)
        black = self._sigmoid(np.zeros(3, dtype=np.float64))
        white = self._sigmoid(np.ones(3, dtype=np.float64))
        derivative = (
            self.response_slopes
            * sigmoid
            * (1.0 - sigmoid)
            / ((layer + self.exposure_floor) * np.log(2.0))
            / (white - black)
        )
        mixed_derivative = (
            1.0 - self.nonlinear_strength
        ) + self.nonlinear_strength * derivative
        determinant = (
            float(np.linalg.det(self.capture_matrix))
            * float(np.linalg.det(self.scan_matrix))
            * np.prod(mixed_derivative, axis=1)
        )
        return determinant.reshape(rgb.shape[:-1])

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "capture_matrix": self.capture_matrix.tolist(),
            "response_midpoints": self.response_midpoints.tolist(),
            "response_slopes": self.response_slopes.tolist(),
            "scan_matrix": self.scan_matrix.tolist(),
            "nonlinear_strength": float(self.nonlinear_strength),
            "identity_derivative_floor": float(1.0 - self.nonlinear_strength),
            "exposure_floor": float(self.exposure_floor),
            "hard_output_clipping": False,
        }


@dataclass(frozen=True)
class IdentityResidualSigmoidFitResult:
    operator: IdentityResidualSigmoidOperator
    development_rgb_rmse: float
    development_maximum_absolute_error: float
    optimization_cost: float
    optimality: float
    function_evaluations: int
    restart_index: int
    converged: bool


def _decode(
    parameters: np.ndarray,
    *,
    matrix_identity_mixture: float,
    nonlinear_strength: float,
) -> IdentityResidualSigmoidOperator:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (18,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must contain eighteen finite values")
    return IdentityResidualSigmoidOperator(
        capture_matrix=row_stochastic_identity_mixture(
            values[:6], identity_mixture=matrix_identity_mixture
        ),
        scan_matrix=row_stochastic_identity_mixture(
            values[6:12], identity_mixture=matrix_identity_mixture
        ),
        response_midpoints=values[12:15],
        response_slopes=values[15:18],
        nonlinear_strength=nonlinear_strength,
    )


def fit_identity_residual_sigmoid(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    nonlinear_strength: float = 0.75,
    matrix_identity_mixture: float = 0.25,
    matrix_logit_bounds: tuple[float, float] = (-8.0, 4.0),
    midpoint_bounds: tuple[float, float] = (-12.0, 2.0),
    slope_bounds: tuple[float, float] = (0.2, 4.0),
    restart_count: int = 2,
    maximum_function_evaluations: int = 2000,
    function_tolerance: float = 1e-10,
    parameter_tolerance: float = 1e-10,
    gradient_tolerance: float = 1e-10,
    loss: PositiveFilmFitLoss = "soft_l1",
    loss_scale: float = 0.02,
    seed: int = 20260801,
) -> IdentityResidualSigmoidFitResult:
    source = _validate_rgb(source_rgb, name="source_rgb")
    target = _validate_rgb(target_rgb, name="target_rgb")
    if source.ndim != 2 or target.shape != source.shape or len(source) < 12:
        raise ValueError("fit expects matching Nx3 arrays with N>=12")
    if restart_count < 1 or maximum_function_evaluations < 1:
        raise ValueError("fit counts must be positive")
    matrix_lower, matrix_upper = map(float, matrix_logit_bounds)
    midpoint_lower, midpoint_upper = map(float, midpoint_bounds)
    slope_lower, slope_upper = map(float, slope_bounds)
    lower = np.concatenate(
        (
            np.full(12, matrix_lower),
            np.full(3, midpoint_lower),
            np.full(3, slope_lower),
        )
    )
    upper = np.concatenate(
        (
            np.full(12, matrix_upper),
            np.full(3, midpoint_upper),
            np.full(3, slope_upper),
        )
    )
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or loss not in ("linear", "soft_l1", "huber", "cauchy", "arctan")
        or not np.isfinite(loss_scale)
        or loss_scale <= 0.0
    ):
        raise ValueError("invalid fit bounds or loss")

    initial = np.concatenate(
        (np.full(12, np.log(1.0 / 8.0)), np.full(3, -3.0), np.ones(3))
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        operator = _decode(
            parameters,
            matrix_identity_mixture=matrix_identity_mixture,
            nonlinear_strength=nonlinear_strength,
        )
        return (operator.apply(source) - target).reshape(-1)

    best = None
    best_index = -1
    for restart_index in range(restart_count):
        start = initial.copy()
        if restart_index:
            rng = np.random.default_rng(seed + restart_index)
            start[:12] += rng.normal(0.0, 0.35, 12)
            start[12:15] += rng.normal(0.0, 0.35, 3)
            start[15:18] += rng.normal(0.0, 0.15, 3)
        result = least_squares(
            residual,
            np.clip(start, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            x_scale="jac",
            loss=loss,
            f_scale=loss_scale,
        )
        if best is None or result.cost < best.cost:
            best = result
            best_index = restart_index
    if best is None:
        raise ValueError("optimizer produced no result")
    operator = _decode(
        best.x,
        matrix_identity_mixture=matrix_identity_mixture,
        nonlinear_strength=nonlinear_strength,
    )
    error = operator.apply(source) - target
    return IdentityResidualSigmoidFitResult(
        operator=operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        development_maximum_absolute_error=float(np.max(np.abs(error))),
        optimization_cost=float(best.cost),
        optimality=float(best.optimality),
        function_evaluations=int(best.nfev),
        restart_index=best_index,
        converged=bool(best.success),
    )


__all__ = [
    "IdentityResidualSigmoidFitResult",
    "IdentityResidualSigmoidOperator",
    "SCHEMA",
    "fit_identity_residual_sigmoid",
]
