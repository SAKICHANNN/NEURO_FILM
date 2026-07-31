"""Strict-interior version of the bounded identity-residual sigmoid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .identity_residual_sigmoid import (
    IdentityResidualSigmoidFitResult,
    IdentityResidualSigmoidOperator,
    _decode,
    _validate_rgb,
)
from .positive_film_fitting import PositiveFilmFitLoss


SCHEMA = "roll2film.strict_interior_identity_residual_sigmoid.v1"


@dataclass(frozen=True)
class StrictInteriorSigmoidOperator:
    base: IdentityResidualSigmoidOperator
    output_epsilon: float = 0.5 / 255.0

    def __post_init__(self) -> None:
        epsilon = float(self.output_epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
            raise ValueError("output_epsilon must lie strictly in (0,0.5)")
        object.__setattr__(self, "output_epsilon", epsilon)

    def apply(self, input_rgb: np.ndarray) -> np.ndarray:
        base_output = self.base.apply(input_rgb)
        return self.output_epsilon + (1.0 - 2.0 * self.output_epsilon) * base_output

    def jacobian_determinants(self, input_rgb: np.ndarray) -> np.ndarray:
        factor = (1.0 - 2.0 * self.output_epsilon) ** 3
        return factor * self.base.jacobian_determinants(input_rgb)

    def to_dict(self) -> dict:
        payload = self.base.to_dict()
        payload["schema"] = SCHEMA
        payload["output_epsilon"] = self.output_epsilon
        payload["strict_interior_parameterization"] = True
        return payload


def fit_strict_interior_sigmoid(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    output_epsilon: float = 0.5 / 255.0,
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

    def decode(parameters: np.ndarray) -> StrictInteriorSigmoidOperator:
        return StrictInteriorSigmoidOperator(
            base=_decode(
                parameters,
                matrix_identity_mixture=matrix_identity_mixture,
                nonlinear_strength=nonlinear_strength,
            ),
            output_epsilon=output_epsilon,
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (decode(parameters).apply(source) - target).reshape(-1)

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
    operator = decode(best.x)
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
    "SCHEMA",
    "StrictInteriorSigmoidOperator",
    "fit_strict_interior_sigmoid",
]
