"""Positive monotone tone base followed by a safe smooth colour residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.roll2film.generalized_monotone_curve_matrix import (
    GeneralizedMonotoneCurveMatrixOperator,
    fit_generalized_monotone_curve_matrix,
)
from src.roll2film.positive_film_fitting import PositiveFilmFitLoss
from src.roll2film.safe_bernstein_lut import (
    SafeBernsteinLUTOperator,
    fit_safe_bernstein_lut,
)


SCHEMA = "roll2film.factorized_monotone_bernstein.v1"


@dataclass(frozen=True)
class FactorizedMonotoneBernsteinOperator:
    base: GeneralizedMonotoneCurveMatrixOperator
    residual: SafeBernsteinLUTOperator

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return self.residual.apply(self.base.apply(rgb))

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        base_output = self.base.apply(rgb)
        return self.base.jacobian_determinants(
            rgb
        ) * self.residual.jacobian_determinants(base_output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "base": self.base.to_dict(),
            "residual": self.residual.to_dict(),
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FactorizedMonotoneBernsteinOperator":
        if (
            payload.get("schema") != SCHEMA
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("unsupported factorized monotone Bernstein schema")
        return cls(
            base=GeneralizedMonotoneCurveMatrixOperator.from_dict(
                payload["base"]
            ),
            residual=SafeBernsteinLUTOperator.from_dict(payload["residual"]),
        )


@dataclass(frozen=True)
class FactorizedMonotoneBernsteinFitResult:
    operator: FactorizedMonotoneBernsteinOperator
    development_rgb_rmse: float
    base_development_rgb_rmse: float
    residual_unconstrained_development_rgb_rmse: float
    minimum_residual_audit_jacobian_determinant: float
    converged: bool


def fit_factorized_monotone_bernstein(
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
    residual_degree: int,
    residual_identity_ridge: float,
    jacobian_floor: float,
    safety_grid_size: int,
    strength_steps: int,
    maximum_residual_iterations: int,
    sample_weights: np.ndarray | None = None,
    residual_strength_cap: float = 1.0,
) -> FactorizedMonotoneBernsteinFitResult:
    cap = float(residual_strength_cap)
    if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise ValueError("residual_strength_cap must be finite in (0, 1]")
    base = fit_generalized_monotone_curve_matrix(
        source,
        target,
        segment_count=segment_count,
        curve_learned_mixture=curve_learned_mixture,
        matrix_identity_mixture=matrix_identity_mixture,
        free_logit_bounds=free_logit_bounds,
        restart_count=restart_count,
        maximum_function_evaluations=maximum_function_evaluations,
        function_tolerance=function_tolerance,
        parameter_tolerance=parameter_tolerance,
        gradient_tolerance=gradient_tolerance,
        loss=loss,
        loss_scale=loss_scale,
        seed=seed,
        sample_weights=sample_weights,
    )
    base_output = base.operator.apply(source)
    residual = fit_safe_bernstein_lut(
        base_output,
        target,
        degree=residual_degree,
        identity_ridge=residual_identity_ridge,
        jacobian_floor=jacobian_floor,
        safety_grid_size=safety_grid_size,
        strength_steps=strength_steps,
        maximum_iterations=maximum_residual_iterations,
        sample_weights=sample_weights,
    )
    residual_operator = residual.operator
    if residual_operator.strength > cap:
        residual_operator = SafeBernsteinLUTOperator(
            degree=residual_operator.degree,
            fitted_control_points=residual_operator.fitted_control_points,
            strength=cap,
            jacobian_floor=residual_operator.jacobian_floor,
        )
    operator = FactorizedMonotoneBernsteinOperator(
        base=base.operator, residual=residual_operator
    )
    error = operator.apply(source) - target
    return FactorizedMonotoneBernsteinFitResult(
        operator=operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        base_development_rgb_rmse=base.development_rgb_rmse,
        residual_unconstrained_development_rgb_rmse=(
            residual.unconstrained_development_rgb_rmse
        ),
        minimum_residual_audit_jacobian_determinant=(
            residual.minimum_audit_jacobian_determinant
        ),
        converged=base.converged and residual.converged,
    )


__all__ = [
    "FactorizedMonotoneBernsteinFitResult",
    "FactorizedMonotoneBernsteinOperator",
    "fit_factorized_monotone_bernstein",
]
