"""Bounded Gaussian residuals for explicitly unidentified RGB code domains."""

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


SCHEMA = "roll2film.code_domain_gaussian_residual.v1"
WORKING_DOMAIN = "unidentified_paired_rgb_code_values"


def _rgb_rows(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data ending in RGB")
    return values.reshape(-1, 3), values.shape


@dataclass(frozen=True)
class CodeDomainGaussianResidualOperator:
    """Fixed-geometry local affine residual with analytical cube headroom."""

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
            or not np.all(np.isfinite(centers))
            or np.any(centers < 0.0)
            or np.any(centers > 1.0)
            or coefficients.shape != (expected, 3)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("invalid code-domain Gaussian parameters")
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

    def residual(self, rgb: np.ndarray) -> np.ndarray:
        rows, shape = _rgb_rows(rgb)
        design = gaussian_affine_design(
            rows,
            self.centers,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        return (design @ self.coefficients).reshape(shape)

    def apply(self, rgb: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        values = np.asarray(rgb, dtype=np.float64)
        _rgb_rows(values)
        if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        if strength == 0.0:
            return values.copy()
        return signed_headroom_map(
            values,
            strength * self.residual(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "working_domain": self.working_domain,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "centers": self.centers.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "CodeDomainGaussianResidualOperator":
        if payload.get("schema") != SCHEMA:
            raise ValueError("unsupported code-domain Gaussian schema")
        return cls(
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            epsilon=float(payload["epsilon"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            working_domain=str(payload["working_domain"]),
        )


def fit_code_domain_gaussian_residual(
    source: np.ndarray,
    target: np.ndarray,
    *,
    axis_size: int,
    sigma: float,
    epsilon: float,
    ridge: float,
) -> CodeDomainGaussianResidualOperator:
    """Fit the explicit residual by deterministic ridge regression."""

    source_rows, _ = _rgb_rows(source)
    target_rows, _ = _rgb_rows(target)
    if source_rows.shape != target_rows.shape:
        raise ValueError("source and target rows must match")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    centers = regular_rgb_centers(axis_size)
    design = gaussian_affine_design(
        source_rows,
        centers,
        sigma=sigma,
        epsilon=epsilon,
    )
    desired = inverse_signed_headroom(source_rows, target_rows)
    gram = design.T @ design
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        design.T @ desired,
    )
    return CodeDomainGaussianResidualOperator(
        centers=centers,
        sigma=sigma,
        epsilon=epsilon,
        coefficients=coefficients,
    )


__all__ = [
    "CodeDomainGaussianResidualOperator",
    "SCHEMA",
    "WORKING_DOMAIN",
    "fit_code_domain_gaussian_residual",
]
