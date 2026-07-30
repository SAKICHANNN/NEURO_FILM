"""Smooth bounded Bernstein LUTs with an explicit Jacobian safety scale."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear


SCHEMA = "roll2film.safe_bernstein_lut.v1"


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


def _bernstein(values: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    indices = np.arange(degree + 1, dtype=np.float64).reshape(1, -1)
    coefficients = np.asarray(
        [comb(degree, index) for index in range(degree + 1)],
        dtype=np.float64,
    ).reshape(1, -1)
    return coefficients * np.power(x, indices) * np.power(
        1.0 - x, degree - indices
    )


def _bernstein_derivative(values: np.ndarray, degree: int) -> np.ndarray:
    if degree < 1:
        return np.zeros((len(values), 1), dtype=np.float64)
    lower = _bernstein(values, degree - 1)
    padded_left = np.pad(lower, ((0, 0), (1, 0)))
    padded_right = np.pad(lower, ((0, 0), (0, 1)))
    return float(degree) * (padded_left - padded_right)


def _design(rgb: np.ndarray, degree: int) -> np.ndarray:
    values = _rgb(rgb).reshape(-1, 3)
    bases = [_bernstein(values[:, channel], degree) for channel in range(3)]
    return np.einsum(
        "ni,nj,nk->nijk", bases[0], bases[1], bases[2]
    ).reshape(len(values), -1)


def _identity_control_points(degree: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, degree + 1)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


@dataclass(frozen=True)
class SafeBernsteinLUTOperator:
    degree: int
    fitted_control_points: np.ndarray
    strength: float
    jacobian_floor: float

    def __post_init__(self) -> None:
        degree = int(self.degree)
        controls = np.asarray(self.fitted_control_points, dtype=np.float64)
        strength = float(self.strength)
        floor = float(self.jacobian_floor)
        expected = ((degree + 1) ** 3, 3)
        if (
            degree < 2
            or controls.shape != expected
            or not np.all(np.isfinite(controls))
            or np.any(controls < 0.0)
            or np.any(controls > 1.0)
            or not 0.0 <= strength <= 1.0
            or not np.isfinite(floor)
            or floor <= 0.0
        ):
            raise ValueError("invalid safe Bernstein LUT contract")
        controls = controls.copy()
        controls.setflags(write=False)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "fitted_control_points", controls)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "jacobian_floor", floor)

    @property
    def control_points(self) -> np.ndarray:
        identity = _identity_control_points(self.degree)
        return identity + self.strength * (
            self.fitted_control_points - identity
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _rgb(rgb)
        output = _design(values, self.degree) @ self.control_points
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise RuntimeError("safe Bernstein LUT escaped cube")
        return np.clip(output, 0.0, 1.0).reshape(values.shape)

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        values = _rgb(rgb)
        flat = values.reshape(-1, 3)
        bases = [_bernstein(flat[:, channel], self.degree) for channel in range(3)]
        derivatives = [
            _bernstein_derivative(flat[:, channel], self.degree)
            for channel in range(3)
        ]
        controls = self.control_points.reshape(
            self.degree + 1, self.degree + 1, self.degree + 1, 3
        )
        jacobian = np.empty((len(flat), 3, 3), dtype=np.float64)
        jacobian[:, :, 0] = np.einsum(
            "ni,nj,nk,ijkc->nc",
            derivatives[0],
            bases[1],
            bases[2],
            controls,
        )
        jacobian[:, :, 1] = np.einsum(
            "ni,nj,nk,ijkc->nc",
            bases[0],
            derivatives[1],
            bases[2],
            controls,
        )
        jacobian[:, :, 2] = np.einsum(
            "ni,nj,nk,ijkc->nc",
            bases[0],
            bases[1],
            derivatives[2],
            controls,
        )
        return np.linalg.det(jacobian).reshape(values.shape[:-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "degree": self.degree,
            "fitted_control_points": self.fitted_control_points.tolist(),
            "strength": self.strength,
            "jacobian_floor": self.jacobian_floor,
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SafeBernsteinLUTOperator":
        if (
            payload.get("schema") != SCHEMA
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("unsupported safe Bernstein LUT schema")
        return cls(
            degree=int(payload["degree"]),
            fitted_control_points=np.asarray(
                payload["fitted_control_points"], dtype=np.float64
            ),
            strength=float(payload["strength"]),
            jacobian_floor=float(payload["jacobian_floor"]),
        )


@dataclass(frozen=True)
class SafeBernsteinFitResult:
    operator: SafeBernsteinLUTOperator
    development_rgb_rmse: float
    unconstrained_development_rgb_rmse: float
    minimum_audit_jacobian_determinant: float
    converged: bool


def fit_safe_bernstein_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    degree: int,
    identity_ridge: float,
    jacobian_floor: float,
    safety_grid_size: int,
    strength_steps: int,
    maximum_iterations: int,
) -> SafeBernsteinFitResult:
    source_values = _rgb(source)
    target_values = _rgb(target)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or degree < 2
        or len(source_values) < (degree + 1) ** 3
        or not np.isfinite(identity_ridge)
        or identity_ridge < 0.0
        or not np.isfinite(jacobian_floor)
        or jacobian_floor <= 0.0
        or safety_grid_size < 5
        or strength_steps < 2
        or maximum_iterations < 1
    ):
        raise ValueError("invalid safe Bernstein LUT fit contract")
    design = _design(source_values, degree)
    identity = _identity_control_points(degree)
    if identity_ridge > 0.0:
        augmented_design = np.concatenate(
            (
                design,
                np.sqrt(identity_ridge)
                * np.eye(len(identity), dtype=np.float64),
            ),
            axis=0,
        )
    else:
        augmented_design = design
    controls = np.empty_like(identity)
    converged = True
    for channel in range(3):
        target_channel = target_values[:, channel]
        if identity_ridge > 0.0:
            target_channel = np.concatenate(
                (
                    target_channel,
                    np.sqrt(identity_ridge) * identity[:, channel],
                )
            )
        result = lsq_linear(
            augmented_design,
            target_channel,
            bounds=(0.0, 1.0),
            method="trf",
            tol=1e-10,
            max_iter=maximum_iterations,
            lsmr_tol="auto",
        )
        controls[:, channel] = result.x
        converged = converged and bool(result.success)
    unconstrained = SafeBernsteinLUTOperator(
        degree=degree,
        fitted_control_points=controls,
        strength=1.0,
        jacobian_floor=jacobian_floor,
    )
    axis = np.linspace(0.0, 1.0, safety_grid_size)
    audit = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    selected = None
    selected_minimum = None
    for strength in np.linspace(1.0, 0.0, strength_steps):
        candidate = SafeBernsteinLUTOperator(
            degree=degree,
            fitted_control_points=controls,
            strength=float(strength),
            jacobian_floor=jacobian_floor,
        )
        minimum = float(np.min(candidate.jacobian_determinants(audit)))
        if minimum >= jacobian_floor:
            selected = candidate
            selected_minimum = minimum
            break
    if selected is None or selected_minimum is None:
        raise RuntimeError("identity failed Bernstein Jacobian safety audit")
    unconstrained_error = unconstrained.apply(source_values) - target_values
    selected_error = selected.apply(source_values) - target_values
    return SafeBernsteinFitResult(
        operator=selected,
        development_rgb_rmse=float(
            np.sqrt(np.mean(np.square(selected_error)))
        ),
        unconstrained_development_rgb_rmse=float(
            np.sqrt(np.mean(np.square(unconstrained_error)))
        ),
        minimum_audit_jacobian_determinant=selected_minimum,
        converged=converged,
    )


__all__ = [
    "SafeBernsteinFitResult",
    "SafeBernsteinLUTOperator",
    "fit_safe_bernstein_lut",
]
