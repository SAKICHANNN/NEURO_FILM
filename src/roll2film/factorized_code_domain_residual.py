"""Safe monotone global base plus bounded local code-domain residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.roll2film.bounded_gaussian_residual import (
    gaussian_affine_design,
    inverse_signed_headroom,
    regular_rgb_centers,
    signed_headroom_map,
)
from src.roll2film.code_domain_gaussian_residual import WORKING_DOMAIN
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurvePositiveMatrixOperator,
)


SCHEMA = "roll2film.factorized_code_domain_gaussian_residual.v1"


def _rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data ending in RGB")
    return values


@dataclass(frozen=True)
class FactorizedCodeDomainResidualOperator:
    """A positive monotone base followed by a local headroom residual."""

    base: MonotoneCurvePositiveMatrixOperator
    centers: np.ndarray
    sigma: float
    coefficients: np.ndarray
    epsilon: float = 1e-12
    working_domain: str = WORKING_DOMAIN

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = 4 * (len(centers) + 1)
        if (
            centers.ndim != 2
            or centers.shape[1] != 3
            or len(centers) == 0
            or np.any(centers < 0.0)
            or np.any(centers > 1.0)
            or not np.all(np.isfinite(centers))
            or coefficients.shape != (expected, 3)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("invalid factorized residual parameters")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.working_domain != WORKING_DOMAIN:
            raise ValueError("unsupported code-domain identity")
        centers = centers.copy()
        coefficients = coefficients.copy()
        centers.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "coefficients", coefficients)

    def apply(self, rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        values = _rgb(rgb)
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        base = self.base.apply(values)
        if strength == 0.0:
            return base
        design = gaussian_affine_design(
            values,
            self.centers,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        residual = (design @ self.coefficients).reshape(values.shape)
        return signed_headroom_map(base, strength * residual)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "working_domain": self.working_domain,
            "base": self.base.to_dict(),
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "centers": self.centers.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FactorizedCodeDomainResidualOperator":
        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported factorized residual schema")
        return cls(
            base=MonotoneCurvePositiveMatrixOperator.from_dict(payload["base"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            epsilon=float(payload["epsilon"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            working_domain=str(payload["working_domain"]),
        )


def fit_factorized_code_domain_residual(
    base: MonotoneCurvePositiveMatrixOperator,
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis_size: int,
    sigma: float,
    epsilon: float,
    ridge: float,
) -> FactorizedCodeDomainResidualOperator:
    source_values = _rgb(source)
    target_values = _rgb(target)
    if source_values.shape != target_values.shape:
        raise ValueError("source and target must match")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    centers = regular_rgb_centers(axis_size)
    design = gaussian_affine_design(
        source_values,
        centers,
        sigma=sigma,
        epsilon=epsilon,
    )
    desired = inverse_signed_headroom(
        base.apply(source_values), target_values
    ).reshape(-1, 3)
    coefficients = np.linalg.solve(
        design.T @ design
        + ridge * np.eye(design.shape[1], dtype=np.float64),
        design.T @ desired,
    )
    return FactorizedCodeDomainResidualOperator(
        base=base,
        centers=centers,
        sigma=sigma,
        epsilon=epsilon,
        coefficients=coefficients,
    )


__all__ = [
    "FactorizedCodeDomainResidualOperator",
    "SCHEMA",
    "fit_factorized_code_domain_residual",
]
