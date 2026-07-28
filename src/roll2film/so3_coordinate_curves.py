"""Bounded SO(3)-coordinate monotone-curve colour operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import torch

from .cube_diffeomorphic_flow import _validate_rgb


SO3_COORDINATE_CURVE_SCHEMA = "roll2film.so3_coordinate_monotone_curve.v1"
POSITIVE_MATRIX_BERNSTEIN_SCHEMA = (
    "roll2film.positive_matrix_bernstein_curve.v1"
)

_DEGREE = 6
_CONTROL_COUNT = _DEGREE + 1
_INTERVAL_COUNT = _DEGREE
_BINOMIAL = np.asarray(
    [comb(_DEGREE, index) for index in range(_CONTROL_COUNT)],
    dtype=np.float64,
)
_CUBE_CORNERS = np.stack(
    np.meshgrid(
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        indexing="ij",
    ),
    axis=-1,
).reshape(-1, 3)


def _validate_rotation_vector(
    value: np.ndarray,
    *,
    maximum_angle_radians: float,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if (
        vector.shape != (3,)
        or not np.all(np.isfinite(vector))
        or not np.isfinite(maximum_angle_radians)
        or maximum_angle_radians <= 0.0
        or float(np.linalg.norm(vector)) > maximum_angle_radians + 1e-12
    ):
        raise ValueError("rotation vector is outside the frozen SO(3) bound")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _validate_controls(
    value: np.ndarray,
    *,
    minimum_increment: float,
) -> np.ndarray:
    controls = np.asarray(value, dtype=np.float64)
    if (
        controls.shape != (3, _CONTROL_COUNT)
        or not np.all(np.isfinite(controls))
        or not np.array_equal(controls[:, 0], np.zeros(3))
        or not np.array_equal(controls[:, -1], np.ones(3))
        or not np.isfinite(minimum_increment)
        or minimum_increment <= 0.0
        or np.any(np.diff(controls, axis=1) < minimum_increment - 1e-12)
    ):
        raise ValueError(
            "controls must have exact endpoints and frozen positive increments"
        )
    result = controls.copy()
    result.setflags(write=False)
    return result


def rotation_matrix_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the right-handed Rodrigues matrix for one bounded rotation."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation_vector must be one finite three-vector")
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def cube_projection_bounds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project all eight RGB-cube corners and return componentwise bounds."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("matrix must be finite 3x3")
    projected = _CUBE_CORNERS @ value
    lower = np.min(projected, axis=0)
    upper = np.max(projected, axis=0)
    if np.any(upper <= lower):
        raise ValueError("projected cube has a degenerate coordinate")
    return lower, upper


def _bernstein_basis_numpy(values: np.ndarray) -> np.ndarray:
    powers = np.arange(_CONTROL_COUNT, dtype=np.int64)
    return (
        _BINOMIAL
        * values[..., None] ** powers
        * (1.0 - values[..., None]) ** (_DEGREE - powers)
    )


def _evaluate_curves_numpy(
    values: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    basis = _bernstein_basis_numpy(values)
    return np.sum(basis * controls, axis=-1)


def _finite_rgb_rows(value: np.ndarray, *, label: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim < 2 or rows.shape[-1] != 3 or not np.all(np.isfinite(rows)):
        raise ValueError(f"{label} must be finite RGB")
    return rows


@dataclass(frozen=True)
class SO3CoordinateCurveOperator:
    """One global SO(3) frame conjugating three monotone scalar curves."""

    rotation_vector: np.ndarray
    control_values: np.ndarray
    strength: float = 1.0
    maximum_rotation_angle_radians: float = 0.45
    minimum_control_increment: float = 0.02
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.strength)
            or self.strength < 0.0
            or self.strength > 1.0
            or self.working_space != "linear_srgb_d65"
        ):
            raise ValueError("unsupported SO(3) coordinate-curve contract")
        vector = _validate_rotation_vector(
            self.rotation_vector,
            maximum_angle_radians=self.maximum_rotation_angle_radians,
        )
        controls = _validate_controls(
            self.control_values,
            minimum_increment=self.minimum_control_increment,
        )
        object.__setattr__(self, "rotation_vector", vector)
        object.__setattr__(self, "control_values", controls)

    @classmethod
    def identity(
        cls,
        *,
        strength: float = 1.0,
        maximum_rotation_angle_radians: float = 0.45,
        minimum_control_increment: float = 0.02,
    ) -> "SO3CoordinateCurveOperator":
        return cls(
            rotation_vector=np.zeros(3, dtype=np.float64),
            control_values=np.broadcast_to(
                np.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=np.float64),
                (3, _CONTROL_COUNT),
            ).copy(),
            strength=strength,
            maximum_rotation_angle_radians=maximum_rotation_angle_radians,
            minimum_control_increment=minimum_control_increment,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_from_vector(self.rotation_vector)

    @property
    def raw_parameter_count(self) -> int:
        return 21

    @property
    def effective_parameter_count(self) -> int:
        return 18

    def _is_exact_identity(self) -> bool:
        identity = np.linspace(
            0.0, 1.0, _CONTROL_COUNT, dtype=np.float64
        )
        return bool(
            self.strength == 0.0
            or (
                np.array_equal(self.rotation_vector, np.zeros(3))
                and np.array_equal(
                    self.control_values,
                    np.broadcast_to(identity, (3, _CONTROL_COUNT)),
                )
            )
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        if self._is_exact_identity():
            return values.copy()
        shape = values.shape
        rows = values.reshape(-1, 3)
        matrix = self.rotation_matrix
        lower, upper = cube_projection_bounds(matrix)
        ranges = upper - lower
        coordinates = rows @ matrix
        normalized = (coordinates - lower) / ranges
        curved = _evaluate_curves_numpy(
            normalized,
            np.broadcast_to(
                self.control_values,
                normalized.shape + (_CONTROL_COUNT,),
            ),
        )
        full = (lower + ranges * curved) @ matrix.T
        result = (1.0 - self.strength) * rows + self.strength * full
        if not np.all(np.isfinite(result)):
            raise RuntimeError("SO(3) coordinate-curve output is non-finite")
        return result.reshape(shape)

    def inverse(
        self,
        rgb: np.ndarray,
        *,
        bisection_iterations: int = 64,
        out_of_curve_range_tolerance: float = 1e-12,
    ) -> np.ndarray:
        target = _finite_rgb_rows(rgb, label="inverse target")
        if bisection_iterations != 64:
            raise ValueError("v1 requires exactly 64 bisection iterations")
        if self.strength == 0.0 or self._is_exact_identity():
            return target.copy()
        if self.strength != 1.0:
            raise ValueError("partial-strength inverse is outside v1")
        shape = target.shape
        rows = target.reshape(-1, 3)
        matrix = self.rotation_matrix
        lower, upper = cube_projection_bounds(matrix)
        ranges = upper - lower
        curved = ((rows @ matrix) - lower) / ranges
        tolerance = float(out_of_curve_range_tolerance)
        if (
            tolerance != 1e-12
            or np.any(curved < -tolerance)
            or np.any(curved > 1.0 + tolerance)
        ):
            raise ValueError("inverse target lies outside the scalar curve range")
        curved = np.where(np.abs(curved) <= tolerance, 0.0, curved)
        curved = np.where(np.abs(curved - 1.0) <= tolerance, 1.0, curved)
        low = np.zeros_like(curved)
        high = np.ones_like(curved)
        controls = np.broadcast_to(
            self.control_values, curved.shape + (_CONTROL_COUNT,)
        )
        for _ in range(bisection_iterations):
            midpoint = (low + high) / 2.0
            value = _evaluate_curves_numpy(midpoint, controls)
            lower_mask = value < curved
            low = np.where(lower_mask, midpoint, low)
            high = np.where(lower_mask, high, midpoint)
        normalized = (low + high) / 2.0
        coordinates = lower + ranges * normalized
        result = coordinates @ matrix.T
        if not np.all(np.isfinite(result)):
            raise RuntimeError("SO(3) coordinate-curve inverse is non-finite")
        return result.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SO3_COORDINATE_CURVE_SCHEMA,
            "working_space": self.working_space,
            "rotation_vector": self.rotation_vector.tolist(),
            "control_values": self.control_values.tolist(),
            "strength": self.strength,
            "maximum_rotation_angle_radians": (
                self.maximum_rotation_angle_radians
            ),
            "minimum_control_increment": self.minimum_control_increment,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SO3CoordinateCurveOperator":
        if payload.get("schema") != SO3_COORDINATE_CURVE_SCHEMA:
            raise ValueError("unsupported SO(3) coordinate-curve schema")
        return cls(
            rotation_vector=np.asarray(
                payload["rotation_vector"], dtype=np.float64
            ),
            control_values=np.asarray(
                payload["control_values"], dtype=np.float64
            ),
            strength=float(payload["strength"]),
            maximum_rotation_angle_radians=float(
                payload["maximum_rotation_angle_radians"]
            ),
            minimum_control_increment=float(
                payload["minimum_control_increment"]
            ),
            working_space=str(payload["working_space"]),
        )


@dataclass(frozen=True)
class PositiveMatrixBernsteinCurveOperator:
    """Three monotone curves followed by a nonnegative stochastic matrix."""

    control_values: np.ndarray
    matrix: np.ndarray
    minimum_control_increment: float = 0.02

    def __post_init__(self) -> None:
        controls = _validate_controls(
            self.control_values,
            minimum_increment=self.minimum_control_increment,
        )
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or np.max(np.abs(np.sum(matrix, axis=1) - 1.0)) > 1e-12
            or float(np.linalg.det(matrix)) <= 0.0
        ):
            raise ValueError("positive matrix must be orientation-preserving")
        result = matrix.copy()
        result.setflags(write=False)
        object.__setattr__(self, "control_values", controls)
        object.__setattr__(self, "matrix", result)

    @classmethod
    def identity(
        cls, *, minimum_control_increment: float = 0.02
    ) -> "PositiveMatrixBernsteinCurveOperator":
        return cls(
            control_values=np.broadcast_to(
                np.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=np.float64),
                (3, _CONTROL_COUNT),
            ).copy(),
            matrix=np.eye(3, dtype=np.float64),
            minimum_control_increment=minimum_control_increment,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        identity = np.linspace(0.0, 1.0, _CONTROL_COUNT)
        if np.array_equal(self.matrix, np.eye(3)) and np.array_equal(
            self.control_values,
            np.broadcast_to(identity, (3, _CONTROL_COUNT)),
        ):
            return values.copy()
        curved = _evaluate_curves_numpy(
            values,
            np.broadcast_to(
                self.control_values, values.shape + (_CONTROL_COUNT,)
            ),
        )
        result = curved @ self.matrix.T
        if not np.all(np.isfinite(result)):
            raise RuntimeError("positive matrix curve output is non-finite")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POSITIVE_MATRIX_BERNSTEIN_SCHEMA,
            "minimum_control_increment": self.minimum_control_increment,
            "control_values": self.control_values.tolist(),
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "PositiveMatrixBernsteinCurveOperator":
        if payload.get("schema") != POSITIVE_MATRIX_BERNSTEIN_SCHEMA:
            raise ValueError("unsupported positive matrix Bernstein schema")
        return cls(
            control_values=np.asarray(
                payload["control_values"], dtype=np.float64
            ),
            matrix=np.asarray(payload["matrix"], dtype=np.float64),
            minimum_control_increment=float(
                payload["minimum_control_increment"]
            ),
        )


def _controls_from_logits_torch(
    logits: torch.Tensor,
    *,
    minimum_increment: float,
) -> torch.Tensor:
    remainder = 1.0 - _INTERVAL_COUNT * minimum_increment
    if remainder <= 0.0:
        raise ValueError("minimum increment is infeasible")
    increments = minimum_increment + remainder * torch.softmax(logits, dim=-1)
    zeros = torch.zeros(
        logits.shape[:-1] + (1,), dtype=logits.dtype, device=logits.device
    )
    return torch.cat((zeros, torch.cumsum(increments, dim=-1)), dim=-1)


def _bernstein_basis_torch(values: torch.Tensor) -> torch.Tensor:
    powers = torch.arange(
        _CONTROL_COUNT, dtype=values.dtype, device=values.device
    )
    binomial = torch.as_tensor(_BINOMIAL, dtype=values.dtype, device=values.device)
    return (
        binomial
        * values[..., None] ** powers
        * (1.0 - values[..., None]) ** (_DEGREE - powers)
    )


def _restart_tensor(
    shape: tuple[int, ...],
    *,
    restart: int,
    seed: int,
    standard_deviation: float,
) -> torch.Tensor:
    if restart == 0:
        return torch.zeros(shape, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + restart)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float64)
        * standard_deviation
    )


def _bounded_rotation_torch(
    raw: torch.Tensor,
    *,
    maximum_angle_radians: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw)
    ratio = torch.tanh(norm) / torch.clamp(norm, min=1e-12)
    ratio = torch.where(norm < 1e-8, 1.0 - norm**2 / 3.0, ratio)
    return maximum_angle_radians * ratio * raw


def _rotation_matrix_torch(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    angle = torch.linalg.vector_norm(vector)
    first = torch.sinc(angle / np.pi)
    second = 0.5 * torch.sinc(angle / (2.0 * np.pi)) ** 2
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + first * skew + second * (skew @ skew)


def _apply_so3_torch(
    rgb: torch.Tensor,
    rotation_vector: torch.Tensor,
    controls: torch.Tensor,
) -> torch.Tensor:
    matrix = _rotation_matrix_torch(rotation_vector)
    corners = torch.as_tensor(
        _CUBE_CORNERS, dtype=rgb.dtype, device=rgb.device
    )
    projected_corners = corners @ matrix
    lower = torch.min(projected_corners, dim=0).values
    upper = torch.max(projected_corners, dim=0).values
    ranges = upper - lower
    normalized = ((rgb @ matrix) - lower) / ranges
    curved = torch.sum(
        _bernstein_basis_torch(normalized) * controls, dim=-1
    )
    return (lower + ranges * curved) @ matrix.T


def fit_so3_coordinate_curve_operator(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    maximum_rotation_angle_radians: float,
    minimum_control_increment: float,
    seed: int,
    restarts: int,
    restart_standard_deviation: float,
    steps: int,
    learning_rate: float,
    rotation_l2: float,
    curve_l2_to_identity: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> tuple[SO3CoordinateCurveOperator, dict[str, Any]]:
    """Fit the frozen bounded coordinate frame and its scalar curves."""

    values = _validate_rgb(rgb).reshape(-1, 3)
    targets = _validate_rgb(target).reshape(-1, 3)
    if targets.shape != values.shape or restarts < 1 or steps < 1:
        raise ValueError("invalid SO(3) coordinate-curve fit inputs")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(thread_count)
    source = torch.from_numpy(values.copy())
    target_tensor = torch.from_numpy(targets.copy())
    identity = torch.linspace(0.0, 1.0, _CONTROL_COUNT, dtype=torch.float64)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    histories: list[dict[str, float | int]] = []
    for restart in range(restarts):
        raw_rotation = torch.nn.Parameter(
            _restart_tensor(
                (3,),
                restart=restart,
                seed=seed,
                standard_deviation=restart_standard_deviation,
            )
        )
        curve_logits = torch.nn.Parameter(
            _restart_tensor(
                (3, _INTERVAL_COUNT),
                restart=restart,
                seed=seed + 1000,
                standard_deviation=restart_standard_deviation,
            )
        )
        optimizer = torch.optim.Adam(
            [raw_rotation, curve_logits], lr=learning_rate
        )
        objective = float("inf")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            rotation = _bounded_rotation_torch(
                raw_rotation,
                maximum_angle_radians=maximum_rotation_angle_radians,
            )
            controls = _controls_from_logits_torch(
                curve_logits,
                minimum_increment=minimum_control_increment,
            )
            prediction = _apply_so3_torch(source, rotation, controls)
            loss = torch.mean((prediction - target_tensor) ** 2)
            loss = loss + rotation_l2 * torch.mean(rotation**2)
            loss = loss + curve_l2_to_identity * torch.mean(
                (controls - identity) ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [raw_rotation, curve_logits], gradient_clip_norm
            )
            optimizer.step()
            objective = float(loss.detach())
        rotation_array = rotation.detach().cpu().numpy()
        control_array = controls.detach().cpu().numpy()
        control_array[:, 0], control_array[:, -1] = 0.0, 1.0
        histories.append({"restart": restart, "final_objective": objective})
        if best is None or (objective, restart) < (best[0], best[1]):
            best = (
                objective,
                restart,
                rotation_array.copy(),
                control_array.copy(),
            )
    assert best is not None
    return (
        SO3CoordinateCurveOperator(
            rotation_vector=best[2],
            control_values=best[3],
            maximum_rotation_angle_radians=maximum_rotation_angle_radians,
            minimum_control_increment=minimum_control_increment,
        ),
        {
            "selected_restart": best[1],
            "objective": best[0],
            "restarts": histories,
        },
    )


def fit_positive_matrix_bernstein_curve_operator(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    minimum_control_increment: float,
    maximum_matrix_mix: float,
    seed: int,
    restarts: int,
    restart_standard_deviation: float,
    steps: int,
    learning_rate: float,
    curve_l2_to_identity: float,
    matrix_l2_to_identity: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> tuple[PositiveMatrixBernsteinCurveOperator, dict[str, Any]]:
    """Fit the frozen compact positive-matrix curve control."""

    values = _validate_rgb(rgb).reshape(-1, 3)
    targets = _validate_rgb(target).reshape(-1, 3)
    if (
        targets.shape != values.shape
        or restarts < 1
        or steps < 1
        or maximum_matrix_mix <= 0.0
        or maximum_matrix_mix >= 1.0
    ):
        raise ValueError("invalid positive matrix curve fit inputs")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(thread_count)
    source = torch.from_numpy(values.copy())
    target_tensor = torch.from_numpy(targets.copy())
    identity_curves = torch.linspace(
        0.0, 1.0, _CONTROL_COUNT, dtype=torch.float64
    )
    identity_matrix = torch.eye(3, dtype=torch.float64)
    basis = _bernstein_basis_torch(source)
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    histories: list[dict[str, float | int]] = []
    for restart in range(restarts):
        curve_logits = torch.nn.Parameter(
            _restart_tensor(
                (3, _INTERVAL_COUNT),
                restart=restart,
                seed=seed + 3000,
                standard_deviation=restart_standard_deviation,
            )
        )
        matrix_logits = torch.nn.Parameter(
            _restart_tensor(
                (3, 3),
                restart=restart,
                seed=seed + 4000,
                standard_deviation=restart_standard_deviation,
            )
        )
        alpha_logit = torch.nn.Parameter(
            torch.tensor(-4.0 + 0.25 * restart, dtype=torch.float64)
        )
        optimizer = torch.optim.Adam(
            [curve_logits, matrix_logits, alpha_logit],
            lr=learning_rate,
        )
        objective = float("inf")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            controls = _controls_from_logits_torch(
                curve_logits,
                minimum_increment=minimum_control_increment,
            )
            alpha = maximum_matrix_mix * torch.sigmoid(alpha_logit)
            matrix = (1.0 - alpha) * identity_matrix + alpha * torch.softmax(
                matrix_logits, dim=1
            )
            curved = torch.sum(basis * controls, dim=-1)
            prediction = curved @ matrix.T
            loss = torch.mean((prediction - target_tensor) ** 2)
            loss = loss + curve_l2_to_identity * torch.mean(
                (controls - identity_curves) ** 2
            )
            loss = loss + matrix_l2_to_identity * torch.mean(
                (matrix - identity_matrix) ** 2
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [curve_logits, matrix_logits, alpha_logit],
                gradient_clip_norm,
            )
            optimizer.step()
            objective = float(loss.detach())
        control_array = controls.detach().cpu().numpy()
        control_array[:, 0], control_array[:, -1] = 0.0, 1.0
        matrix_array = matrix.detach().cpu().numpy()
        histories.append({"restart": restart, "final_objective": objective})
        if best is None or (objective, restart) < (best[0], best[1]):
            best = (
                objective,
                restart,
                control_array.copy(),
                matrix_array.copy(),
            )
    assert best is not None
    return (
        PositiveMatrixBernsteinCurveOperator(
            control_values=best[2],
            matrix=best[3],
            minimum_control_increment=minimum_control_increment,
        ),
        {
            "selected_restart": best[1],
            "objective": best[0],
            "restarts": histories,
        },
    )


__all__ = [
    "POSITIVE_MATRIX_BERNSTEIN_SCHEMA",
    "SO3_COORDINATE_CURVE_SCHEMA",
    "PositiveMatrixBernsteinCurveOperator",
    "SO3CoordinateCurveOperator",
    "cube_projection_bounds",
    "fit_positive_matrix_bernstein_curve_operator",
    "fit_so3_coordinate_curve_operator",
    "rotation_matrix_from_vector",
]
