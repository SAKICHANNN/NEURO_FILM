"""Strict-interior parallel projection-curve colour operators.

This is a clean-room, safety-constrained research family motivated by the
many-projection representation described in arXiv:2510.02713.  It does not
copy that repository's implementation and does not use the term ``pigment``
as a physical film claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PROJECTION_CURVE_SCHEMA = "roll2film.parallel_projection_curves.v1"


class ProjectionCurveError(ValueError):
    """Raised when a projection-curve operator violates its contract."""


def validate_projection_directions(value: np.ndarray) -> np.ndarray:
    directions = np.asarray(value, dtype=np.float64)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or len(directions) < 3
        or not np.all(np.isfinite(directions))
        or np.any(directions < 0.0)
        or np.max(np.abs(np.sum(directions, axis=1) - 1.0)) > 1e-12
    ):
        raise ProjectionCurveError(
            "projection directions must be finite nonnegative simplex rows"
        )
    result = np.ascontiguousarray(directions)
    result.setflags(write=False)
    return result


def _validate_rgb(value: np.ndarray, *, label: str) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)
    if (
        rgb.ndim < 2
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ProjectionCurveError(f"{label} must be finite RGB in [0,1]")
    return rgb


def _hat_design(projections: np.ndarray, control_point_count: int) -> np.ndarray:
    values = np.asarray(projections, dtype=np.float64)
    if (
        values.ndim != 2
        or control_point_count < 3
        or not np.all(np.isfinite(values))
        or np.any(values < -1e-12)
        or np.any(values > 1.0 + 1e-12)
    ):
        raise ProjectionCurveError("invalid projected colours")
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (control_point_count - 1)
    lower = np.minimum(
        np.floor(scaled).astype(np.int64), control_point_count - 2
    )
    fraction = scaled - lower
    rows, directions = values.shape
    design = np.zeros(
        (rows, directions * control_point_count), dtype=np.float64
    )
    row_index = np.arange(rows)[:, None]
    direction_offset = (
        np.arange(directions, dtype=np.int64)[None, :] * control_point_count
    )
    np.add.at(
        design,
        (np.broadcast_to(row_index, lower.shape), direction_offset + lower),
        1.0 - fraction,
    )
    np.add.at(
        design,
        (
            np.broadcast_to(row_index, lower.shape),
            direction_offset + lower + 1,
        ),
        fraction,
    )
    return design


def _difference_matrix(
    direction_count: int, control_point_count: int
) -> np.ndarray:
    row_count = direction_count * (control_point_count - 1)
    column_count = direction_count * control_point_count
    result = np.zeros((row_count, column_count), dtype=np.float64)
    row = 0
    for direction in range(direction_count):
        offset = direction * control_point_count
        for knot in range(control_point_count - 1):
            result[row, offset + knot] = -1.0
            result[row, offset + knot + 1] = 1.0
            row += 1
    return result


def _headroom(rgb: np.ndarray, boundary_epsilon: float) -> np.ndarray:
    return np.maximum(
        0.0,
        np.minimum(rgb - boundary_epsilon, 1.0 - boundary_epsilon - rgb),
    )


@dataclass(frozen=True)
class ParallelProjectionCurveOperator:
    """A deterministic global point operator with intrinsic cube headroom."""

    directions: np.ndarray
    coefficients: np.ndarray
    boundary_epsilon: float
    dose: float = 1.0
    strict_interior_safety_factor: float = 1.0 - 1.0e-10
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        directions = validate_projection_directions(self.directions)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            coefficients.ndim != 3
            or coefficients.shape[0] != len(directions)
            or coefficients.shape[1] < 3
            or coefficients.shape[2] != 3
            or not np.all(np.isfinite(coefficients))
            or not 0.0 < self.boundary_epsilon < 0.5
            or not np.isfinite(self.dose)
            or not 0.0 <= self.dose <= 1.0
            or not 0.0 < self.strict_interior_safety_factor < 1.0
            or self.working_space != "linear_srgb_d65"
        ):
            raise ProjectionCurveError("invalid projection-curve operator")
        frozen_coefficients = np.ascontiguousarray(coefficients)
        frozen_coefficients.setflags(write=False)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "coefficients", frozen_coefficients)

    @property
    def control_point_count(self) -> int:
        return int(self.coefficients.shape[1])

    @classmethod
    def identity(
        cls,
        directions: np.ndarray,
        *,
        control_point_count: int,
        boundary_epsilon: float,
    ) -> "ParallelProjectionCurveOperator":
        validated = validate_projection_directions(directions)
        return cls(
            validated,
            np.zeros(
                (len(validated), control_point_count, 3), dtype=np.float64
            ),
            boundary_epsilon,
        )

    def _apply_rows(self, rows: np.ndarray) -> np.ndarray:
        if self.dose == 0.0 or not np.any(self.coefficients):
            return rows.copy()
        projected = rows @ self.directions.T
        scaled = np.clip(projected, 0.0, 1.0) * (
            self.control_point_count - 1
        )
        lower = np.minimum(
            np.floor(scaled).astype(np.int64), self.control_point_count - 2
        )
        fraction = scaled - lower
        latent = np.zeros((len(rows), 3), dtype=np.float64)
        indices = np.arange(len(rows))
        for direction in range(len(self.directions)):
            controls = self.coefficients[direction]
            lo = controls[lower[:, direction]]
            hi = controls[lower[:, direction] + 1]
            latent += lo + fraction[:, direction, None] * (hi - lo)
        activation = (
            self.strict_interior_safety_factor
            * np.tanh(self.dose * latent)
        )
        result = rows + _headroom(rows, self.boundary_epsilon) * activation
        if (
            not np.all(np.isfinite(result))
            or np.any(result < -1e-12)
            or np.any(result > 1.0 + 1e-12)
        ):
            raise RuntimeError("projection-curve output escaped the RGB cube")
        return result

    def apply(self, rgb: np.ndarray, *, chunk_rows: int = 65536) -> np.ndarray:
        values = _validate_rgb(rgb, label="projection-curve input")
        if chunk_rows < 1:
            raise ProjectionCurveError("chunk_rows must be positive")
        shape = values.shape
        rows = values.reshape(-1, 3)
        output = np.empty_like(rows)
        for start in range(0, len(rows), chunk_rows):
            stop = min(start + chunk_rows, len(rows))
            output[start:stop] = self._apply_rows(rows[start:stop])
        return output.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROJECTION_CURVE_SCHEMA,
            "working_space": self.working_space,
            "directions": self.directions.tolist(),
            "coefficients": self.coefficients.tolist(),
            "boundary_epsilon": self.boundary_epsilon,
            "dose": self.dose,
            "strict_interior_safety_factor": (
                self.strict_interior_safety_factor
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParallelProjectionCurveOperator":
        if payload.get("schema") != PROJECTION_CURVE_SCHEMA:
            raise ProjectionCurveError("unsupported projection-curve schema")
        return cls(
            np.asarray(payload["directions"], dtype=np.float64),
            np.asarray(payload["coefficients"], dtype=np.float64),
            float(payload["boundary_epsilon"]),
            float(payload["dose"]),
            float(payload["strict_interior_safety_factor"]),
            str(payload["working_space"]),
        )


def fit_projection_curve_coefficients(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    directions: np.ndarray,
    control_point_count: int,
    boundary_epsilon: float,
    sample_stride: int,
    target_activation_limit: float,
    identity_shrinkage: float,
    first_difference_smoothness: float,
    coefficient_absolute_limit: float,
    strict_interior_safety_factor: float = 1.0 - 1.0e-10,
) -> np.ndarray:
    source = _validate_rgb(source_rgb, label="fit source")
    target = _validate_rgb(target_rgb, label="fit target")
    projection_directions = validate_projection_directions(directions)
    if (
        source.shape != target.shape
        or source.ndim != 3
        or sample_stride < 1
        or control_point_count < 3
        or not 0.0 < boundary_epsilon < 0.5
        or not 0.0 < target_activation_limit < 1.0
        or identity_shrinkage <= 0.0
        or first_difference_smoothness < 0.0
        or coefficient_absolute_limit <= 0.0
        or not 0.0 < strict_interior_safety_factor < 1.0
    ):
        raise ProjectionCurveError("invalid projection-curve fit contract")
    sampled_source = source[::sample_stride, ::sample_stride].reshape(-1, 3)
    sampled_target = target[::sample_stride, ::sample_stride].reshape(-1, 3)
    headroom = _headroom(sampled_source, boundary_epsilon)
    valid = np.all(headroom > 1e-12, axis=1)
    if int(np.sum(valid)) < control_point_count * len(projection_directions):
        raise ProjectionCurveError("insufficient strict-interior fit samples")
    fit_source = sampled_source[valid]
    fit_target = sampled_target[valid]
    fit_headroom = headroom[valid]
    desired = np.clip(
        (fit_target - fit_source) / fit_headroom,
        -target_activation_limit,
        target_activation_limit,
    )
    latent_target = np.arctanh(desired)
    design = _hat_design(
        fit_source @ projection_directions.T, control_point_count
    )
    differences = _difference_matrix(
        len(projection_directions), control_point_count
    )
    system = (
        design.T @ design
        + identity_shrinkage * np.eye(design.shape[1], dtype=np.float64)
        + first_difference_smoothness * (differences.T @ differences)
    )
    coefficients = np.linalg.solve(system, design.T @ latent_target)
    coefficients = np.clip(
        coefficients,
        -coefficient_absolute_limit,
        coefficient_absolute_limit,
    )
    return coefficients.reshape(
        len(projection_directions), control_point_count, 3
    )


def jacobian_diagnostics(
    operator: ParallelProjectionCurveOperator,
    *,
    grid_size: int,
    finite_difference: float,
) -> dict[str, float | int]:
    if grid_size < 3 or not 0.0 < finite_difference < 0.01:
        raise ProjectionCurveError("invalid Jacobian audit grid")
    margin = operator.boundary_epsilon + 4.0 * finite_difference
    axis = np.linspace(margin, 1.0 - margin, grid_size, dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    rows = points.reshape(-1, 3)
    jacobian = np.empty((len(rows), 3, 3), dtype=np.float64)
    for input_channel in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[input_channel] = finite_difference
        plus = operator.apply(rows + delta)
        minus = operator.apply(rows - delta)
        jacobian[:, :, input_channel] = (
            plus - minus
        ) / (2.0 * finite_difference)
    determinant = np.linalg.det(jacobian)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = singular[:, 0] / np.maximum(singular[:, -1], 1e-15)
    return {
        "sample_count": int(len(rows)),
        "minimum_determinant": float(np.min(determinant)),
        "nonpositive_determinant_count": int(np.sum(determinant <= 0.0)),
        "maximum_condition": float(np.max(condition)),
    }


def select_safe_dose(
    *,
    directions: np.ndarray,
    coefficients: np.ndarray,
    boundary_epsilon: float,
    grid_size: int,
    finite_difference: float,
    minimum_jacobian_determinant: float,
    maximum_jacobian_condition: float,
    bisection_iterations: int,
    strict_interior_safety_factor: float = 1.0 - 1.0e-10,
) -> tuple[ParallelProjectionCurveOperator, dict[str, float | int]]:
    if (
        not 0.0 < minimum_jacobian_determinant < 1.0
        or maximum_jacobian_condition <= 1.0
        or bisection_iterations < 1
    ):
        raise ProjectionCurveError("invalid safe-dose contract")

    def evaluate(dose: float) -> tuple[ParallelProjectionCurveOperator, dict[str, float | int], bool]:
        candidate = ParallelProjectionCurveOperator(
            directions,
            coefficients,
            boundary_epsilon,
            dose,
            strict_interior_safety_factor,
        )
        diagnostics = jacobian_diagnostics(
            candidate,
            grid_size=grid_size,
            finite_difference=finite_difference,
        )
        passed = bool(
            diagnostics["nonpositive_determinant_count"] == 0
            and diagnostics["minimum_determinant"]
            >= minimum_jacobian_determinant
            and diagnostics["maximum_condition"] <= maximum_jacobian_condition
        )
        return candidate, diagnostics, passed

    full, full_diagnostics, full_passed = evaluate(1.0)
    if full_passed:
        return full, full_diagnostics
    low = 0.0
    high = 1.0
    best, best_diagnostics, _ = evaluate(low)
    for _ in range(bisection_iterations):
        middle = (low + high) / 2.0
        candidate, diagnostics, passed = evaluate(middle)
        if passed:
            low = middle
            best = candidate
            best_diagnostics = diagnostics
        else:
            high = middle
    return best, best_diagnostics


__all__ = [
    "PROJECTION_CURVE_SCHEMA",
    "ParallelProjectionCurveOperator",
    "ProjectionCurveError",
    "fit_projection_curve_coefficients",
    "jacobian_diagnostics",
    "select_safe_dose",
    "validate_projection_directions",
]
