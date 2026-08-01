"""Cube-safe luma-conditioned positive RGB mixing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares


PARAMETER_COUNT = 18


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError("RGB must be finite and end in three channels")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("RGB must remain inside the unit cube")
    return value


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _matrices(parameters: np.ndarray, maximum_off_diagonal: float) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    maximum = float(maximum_off_diagonal)
    if values.shape != (PARAMETER_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("matrix operator requires eighteen finite parameters")
    if not math.isfinite(maximum) or maximum <= 0.0 or maximum > 1.0:
        raise ValueError("invalid off-diagonal maximum")
    off = maximum * _sigmoid(values).reshape(3, 3, 2)
    matrices = np.zeros((3, 3, 3), dtype=np.float64)
    for anchor in range(3):
        for row in range(3):
            columns = [column for column in range(3) if column != row]
            matrices[anchor, row, row] = 1.0
            matrices[anchor, row, columns] = off[anchor, row]
            matrices[anchor, row] /= np.sum(matrices[anchor, row])
    return matrices


def _bernstein_weights(luma: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(luma, dtype=np.float64), 0.0, 1.0)
    return np.stack((np.square(1.0 - value), 2.0 * value * (1.0 - value), np.square(value)), axis=-1)


@dataclass(frozen=True)
class LumaConditionedPositiveMatrix:
    parameters: np.ndarray
    maximum_off_diagonal: float

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64).copy()
        _matrices(parameters, self.maximum_off_diagonal)
        parameters.setflags(write=False)
        object.__setattr__(self, "parameters", parameters)

    @property
    def matrices(self) -> np.ndarray:
        return _matrices(self.parameters, self.maximum_off_diagonal)

    def apply(self, rgb: np.ndarray, *, conditioning_rgb: np.ndarray | None = None) -> np.ndarray:
        source = _validate_rgb(rgb)
        conditioning = source if conditioning_rgb is None else _validate_rgb(conditioning_rgb)
        if conditioning.shape != source.shape:
            raise ValueError("conditioning RGB shape mismatch")
        luma = conditioning[..., 0] * 0.2126 + conditioning[..., 1] * 0.7152 + conditioning[..., 2] * 0.0722
        weights = _bernstein_weights(luma)
        anchor_outputs = np.stack(
            [np.einsum("...c,rc->...r", source, matrix) for matrix in self.matrices],
            axis=-2,
        )
        output = np.sum(anchor_outputs * weights[..., :, None], axis=-2)
        if not np.all(np.isfinite(output)) or np.any(output < -1e-12) or np.any(output > 1.0 + 1e-12):
            raise RuntimeError("positive matrix blend escaped RGB cube")
        return np.clip(output, 0.0, 1.0)


def fit_luma_conditioned_positive_matrix(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_off_diagonal: float,
    logit_bounds: tuple[float, float],
    initial_logit: float,
    identity_shrinkage: float,
    maximum_fit_samples: int,
    maximum_evaluations: int,
    loss: str,
    loss_scale: float,
) -> tuple[LumaConditionedPositiveMatrix, bool]:
    x = _validate_rgb(source)
    y = _validate_rgb(target)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("fit arrays must be matching Nx3 RGB")
    if len(x) > maximum_fit_samples:
        indices = np.linspace(0, len(x) - 1, maximum_fit_samples, dtype=np.int64)
        x_fit = x[indices]
        y_fit = y[indices]
    else:
        x_fit, y_fit = x, y
    initial = np.full(PARAMETER_COUNT, float(initial_logit), dtype=np.float64)

    def residual(parameters: np.ndarray) -> np.ndarray:
        operator = LumaConditionedPositiveMatrix(parameters, maximum_off_diagonal)
        prediction = operator.apply(x_fit, conditioning_rgb=x_fit)
        off = maximum_off_diagonal * _sigmoid(parameters)
        return np.concatenate(((prediction - y_fit).ravel(), np.sqrt(identity_shrinkage) * off))

    result = least_squares(
        residual,
        initial,
        bounds=(float(logit_bounds[0]), float(logit_bounds[1])),
        max_nfev=int(maximum_evaluations),
        loss=str(loss),
        f_scale=float(loss_scale),
    )
    return LumaConditionedPositiveMatrix(result.x, maximum_off_diagonal), bool(result.success)


__all__ = ["LumaConditionedPositiveMatrix", "fit_luma_conditioned_positive_matrix"]
