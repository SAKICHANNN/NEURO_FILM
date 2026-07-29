"""Compact neutral-preserving Gaussian residuals for explicit colour operators.

The representation is a conservative clean-room subset of Gaussian LUTs:
Gaussian geometry is fixed independently of the fitted pairs, and each
primitive contributes only a local RGB bias on top of an already bounded
global operator.  Subtracting the same basis evaluated on the input's neutral
projection makes the added residual exactly zero on the neutral axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .positive_film import PositiveFilmResponseOperator


FIXED_GAUSSIAN_RESIDUAL_SCHEMA = (
    "roll2film.fixed_neutral_gaussian_residual.v1"
)
FIXED_GAUSSIAN_LOG_ODDS_SCHEMA = (
    "roll2film.fixed_neutral_gaussian_log_odds.v1"
)


def fixed_cube_centers(levels: tuple[float, ...]) -> np.ndarray:
    """Return a deterministic Cartesian RGB centre grid."""

    axis = np.asarray(levels, dtype=np.float64)
    if (
        axis.ndim != 1
        or len(axis) < 2
        or not np.all(np.isfinite(axis))
        or np.any(axis <= 0.0)
        or np.any(axis >= 1.0)
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError("centre levels must be strictly increasing inside (0, 1)")
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _rgb_rows(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data ending in three channels")
    return values.reshape(-1, 3), values.shape


@dataclass(frozen=True)
class FixedNeutralGaussianResidualOperator:
    """A bounded-global base plus a continuous local Gaussian residual."""

    base: PositiveFilmResponseOperator
    centers: np.ndarray
    sigma: float
    coefficients: np.ndarray
    normalization_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            centers.ndim != 2
            or centers.shape[1] != 3
            or len(centers) < 2
            or coefficients.shape != centers.shape
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(coefficients))
            or np.any(centers <= 0.0)
            or np.any(centers >= 1.0)
        ):
            raise ValueError("centres and coefficients must be finite matching Nx3")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if (
            not np.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0.0
        ):
            raise ValueError("normalization_epsilon must be finite and positive")
        centers = centers.copy()
        coefficients = coefficients.copy()
        centers.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "coefficients", coefficients)

    def _weights(self, rows: np.ndarray) -> np.ndarray:
        residual = rows[:, None, :] - self.centers[None, :, :]
        log_weight = -0.5 * np.sum(residual * residual, axis=2) / (
            self.sigma * self.sigma
        )
        maximum = np.max(log_weight, axis=1, keepdims=True)
        weight = np.exp(log_weight - maximum)
        return weight / (
            np.sum(weight, axis=1, keepdims=True)
            + self.normalization_epsilon
        )

    def features(self, rgb: np.ndarray) -> np.ndarray:
        """Return basis features that vanish exactly for neutral RGB."""

        rows, _ = _rgb_rows(rgb)
        neutral_level = np.mean(rows, axis=1, keepdims=True)
        neutral = np.repeat(neutral_level, 3, axis=1)
        features = self._weights(rows) - self._weights(neutral)
        neutral_rows = np.max(rows, axis=1) == np.min(rows, axis=1)
        features[neutral_rows] = 0.0
        return features

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        rows, shape = _rgb_rows(rgb)
        base = self.base.apply(rows)
        output = base + self.features(rows) @ self.coefficients
        if not np.all(np.isfinite(output)):
            raise RuntimeError("Gaussian residual produced non-finite output")
        return output.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_GAUSSIAN_RESIDUAL_SCHEMA,
            "base": self.base.to_dict(),
            "centers": self.centers.tolist(),
            "sigma": self.sigma,
            "coefficients": self.coefficients.tolist(),
            "normalization_epsilon": self.normalization_epsilon,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FixedNeutralGaussianResidualOperator":
        if payload.get("schema") != FIXED_GAUSSIAN_RESIDUAL_SCHEMA:
            raise ValueError("unsupported fixed Gaussian residual schema")
        return cls(
            base=PositiveFilmResponseOperator.from_dict(payload["base"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            normalization_epsilon=float(payload["normalization_epsilon"]),
        )


@dataclass(frozen=True)
class FixedNeutralGaussianLogOddsOperator:
    """A cube-preserving Gaussian residual in output-channel log-odds."""

    base: PositiveFilmResponseOperator
    centers: np.ndarray
    sigma: float
    coefficients: np.ndarray
    normalization_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        # Reuse the additive operator's strict immutable parameter validation.
        validated = FixedNeutralGaussianResidualOperator(
            base=self.base,
            centers=self.centers,
            sigma=self.sigma,
            coefficients=self.coefficients,
            normalization_epsilon=self.normalization_epsilon,
        )
        object.__setattr__(self, "centers", validated.centers)
        object.__setattr__(self, "coefficients", validated.coefficients)

    def features(self, rgb: np.ndarray) -> np.ndarray:
        helper = FixedNeutralGaussianResidualOperator(
            base=self.base,
            centers=self.centers,
            sigma=self.sigma,
            coefficients=self.coefficients,
            normalization_epsilon=self.normalization_epsilon,
        )
        return helper.features(rgb)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        rows, shape = _rgb_rows(rgb)
        base = self.base.apply(rows)
        delta = self.features(rows) @ self.coefficients
        # This is sigmoid(logit(base) + delta), written without evaluating
        # logit(0) or logit(1). The half-exponent form is stable under the
        # finite fitted coefficients allowed by the constructor.
        positive = np.exp(np.clip(0.5 * delta, -350.0, 350.0))
        negative = np.exp(np.clip(-0.5 * delta, -350.0, 350.0))
        numerator = base * positive
        denominator = numerator + (1.0 - base) * negative
        output = numerator / denominator
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("Gaussian log-odds residual escaped the RGB cube")
        return output.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_GAUSSIAN_LOG_ODDS_SCHEMA,
            "base": self.base.to_dict(),
            "centers": self.centers.tolist(),
            "sigma": self.sigma,
            "coefficients": self.coefficients.tolist(),
            "normalization_epsilon": self.normalization_epsilon,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "FixedNeutralGaussianLogOddsOperator":
        if payload.get("schema") != FIXED_GAUSSIAN_LOG_ODDS_SCHEMA:
            raise ValueError("unsupported fixed Gaussian log-odds schema")
        return cls(
            base=PositiveFilmResponseOperator.from_dict(payload["base"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            normalization_epsilon=float(payload["normalization_epsilon"]),
        )


def fit_fixed_neutral_gaussian_residual(
    base: PositiveFilmResponseOperator,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    centers: np.ndarray,
    sigma: float,
    ridge: float,
) -> FixedNeutralGaussianResidualOperator:
    """Fit only local-bias coefficients by deterministic ridge regression."""

    source, _ = _rgb_rows(source_rgb)
    target, _ = _rgb_rows(target_rgb)
    if target.shape != source.shape or len(source) < len(centers):
        raise ValueError("paired rows must match and cover every Gaussian centre")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    zero = np.zeros_like(np.asarray(centers, dtype=np.float64))
    provisional = FixedNeutralGaussianResidualOperator(
        base=base,
        centers=centers,
        sigma=sigma,
        coefficients=zero,
    )
    features = provisional.features(source)
    residual = target - base.apply(source)
    gram = features.T @ features + ridge * np.eye(
        features.shape[1], dtype=np.float64
    )
    coefficients = np.linalg.solve(gram, features.T @ residual)
    return FixedNeutralGaussianResidualOperator(
        base=base,
        centers=centers,
        sigma=sigma,
        coefficients=coefficients,
    )


def fit_fixed_neutral_gaussian_log_odds(
    base: PositiveFilmResponseOperator,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    centers: np.ndarray,
    sigma: float,
    ridge: float,
    fit_epsilon: float = 1e-6,
) -> FixedNeutralGaussianLogOddsOperator:
    """Fit a deterministic cube-preserving local residual in log-odds."""

    source, _ = _rgb_rows(source_rgb)
    target, _ = _rgb_rows(target_rgb)
    if target.shape != source.shape or len(source) < len(centers):
        raise ValueError("paired rows must match and cover every Gaussian centre")
    if (
        not np.isfinite(ridge)
        or ridge <= 0.0
        or not np.isfinite(fit_epsilon)
        or fit_epsilon <= 0.0
        or fit_epsilon >= 0.5
    ):
        raise ValueError("ridge and fit_epsilon are invalid")
    zero = np.zeros_like(np.asarray(centers, dtype=np.float64))
    provisional = FixedNeutralGaussianLogOddsOperator(
        base=base,
        centers=centers,
        sigma=sigma,
        coefficients=zero,
    )
    features = provisional.features(source)
    base_output = np.clip(base.apply(source), fit_epsilon, 1.0 - fit_epsilon)
    target_output = np.clip(target, fit_epsilon, 1.0 - fit_epsilon)
    residual = (
        np.log(target_output) - np.log1p(-target_output)
        - np.log(base_output)
        + np.log1p(-base_output)
    )
    gram = features.T @ features + ridge * np.eye(
        features.shape[1], dtype=np.float64
    )
    coefficients = np.linalg.solve(gram, features.T @ residual)
    return FixedNeutralGaussianLogOddsOperator(
        base=base,
        centers=centers,
        sigma=sigma,
        coefficients=coefficients,
    )


__all__ = [
    "FIXED_GAUSSIAN_LOG_ODDS_SCHEMA",
    "FIXED_GAUSSIAN_RESIDUAL_SCHEMA",
    "FixedNeutralGaussianLogOddsOperator",
    "FixedNeutralGaussianResidualOperator",
    "fit_fixed_neutral_gaussian_log_odds",
    "fit_fixed_neutral_gaussian_residual",
    "fixed_cube_centers",
]
