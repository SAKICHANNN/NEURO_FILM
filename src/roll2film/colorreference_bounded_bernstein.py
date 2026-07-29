"""Analytically bounded Bernstein-lattice recorder proxy."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class BoundedBernsteinModel:
    degree: int
    coefficients: np.ndarray
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        expected = (self.degree + 1) ** 3
        values = np.asarray(self.coefficients, dtype=np.float64)
        if (
            self.degree < 1
            or values.shape != (expected, 3)
            or not np.all(np.isfinite(values))
            or not np.isfinite(self.lower_bound)
            or not np.isfinite(self.upper_bound)
            or self.lower_bound >= self.upper_bound
            or np.any(values < self.lower_bound)
            or np.any(values > self.upper_bound)
        ):
            raise ValueError("invalid bounded Bernstein model")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)

    @property
    def model_id(self) -> str:
        return f"bernstein_d{self.degree}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "degree": self.degree,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "coefficients": self.coefficients.tolist(),
        }


def bernstein_tensor_basis(
    source: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Return lexicographic tensor Bernstein weights on [0,1]^3."""

    values = np.asarray(source, dtype=np.float64)
    if (
        degree < 1
        or values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("Bernstein source must be finite Nx3 in [0,1]")
    channel_bases = []
    for channel in range(3):
        x = values[:, channel]
        channel_bases.append(
            np.column_stack(
                [
                    comb(degree, index)
                    * np.power(x, index)
                    * np.power(1.0 - x, degree - index)
                    for index in range(degree + 1)
                ]
            )
        )
    return np.einsum(
        "ni,nj,nk->nijk",
        channel_bases[0],
        channel_bases[1],
        channel_bases[2],
        optimize=True,
    ).reshape(values.shape[0], -1)


def fit_bounded_bernstein(
    source: np.ndarray,
    target_xyz: np.ndarray,
    *,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
    maximum_iterations: int,
) -> BoundedBernsteinModel:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target_xyz, dtype=np.float64)
    if (
        y.shape != x.shape
        or x.shape[0] < (degree + 1) ** 3
        or not np.all(np.isfinite(y))
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or maximum_iterations < 1
    ):
        raise ValueError("invalid bounded Bernstein fit data")
    basis = bernstein_tensor_basis(x, degree=degree)
    columns = []
    for channel in range(3):
        result = lsq_linear(
            basis,
            y[:, channel],
            bounds=(lower_bound, upper_bound),
            method="trf",
            tol=tolerance,
            lsmr_tol="auto",
            max_iter=maximum_iterations,
            verbose=0,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError("bounded Bernstein fit did not converge")
        columns.append(result.x)
    return BoundedBernsteinModel(
        degree=degree,
        coefficients=np.stack(columns, axis=1),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def apply_bounded_bernstein(
    model: BoundedBernsteinModel,
    source: np.ndarray,
) -> np.ndarray:
    basis = bernstein_tensor_basis(source, degree=model.degree)
    output = basis @ model.coefficients
    tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        np.any(output < model.lower_bound - tolerance)
        or np.any(output > model.upper_bound + tolerance)
    ):
        raise RuntimeError("Bernstein analytic bound invariant failed")
    return output
