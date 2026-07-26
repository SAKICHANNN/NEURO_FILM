"""Bounded explicit colour operators from cube-preserving stationary flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


CUBE_DIFFEO_FLOW_SCHEMA = "roll2film.cube_diffeomorphic_colour_flow.v1"


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if (
        values.ndim < 2
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("rgb must be finite [0, 1] data with shape (..., 3)")
    return values


def _validate_velocity_grid(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[0] != values.shape[1]
        or values.shape[1] != values.shape[2]
        or values.shape[3] != 3
        or values.shape[0] < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("velocity_grid must have finite shape (K, K, K, 3), K>=2")
    return values


def _cell_coordinates_numpy(
    rgb: np.ndarray, axis_size: int
) -> tuple[np.ndarray, np.ndarray]:
    scaled = rgb * float(axis_size - 1)
    lower = np.floor(scaled).astype(np.int64)
    lower = np.minimum(lower, axis_size - 2)
    fraction = scaled - lower
    return lower, fraction


def _sample_grid_numpy(rgb: np.ndarray, grid: np.ndarray) -> np.ndarray:
    axis_size = grid.shape[0]
    lower, fraction = _cell_coordinates_numpy(rgb, axis_size)
    result = np.zeros_like(rgb)
    for red in (0, 1):
        wr = fraction[:, 0] if red else 1.0 - fraction[:, 0]
        ir = lower[:, 0] + red
        for green in (0, 1):
            wg = fraction[:, 1] if green else 1.0 - fraction[:, 1]
            ig = lower[:, 1] + green
            for blue in (0, 1):
                wb = fraction[:, 2] if blue else 1.0 - fraction[:, 2]
                ib = lower[:, 2] + blue
                result += (wr * wg * wb)[:, None] * grid[ir, ig, ib]
    return result


def _velocity_numpy(rgb: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return rgb * (1.0 - rgb) * _sample_grid_numpy(rgb, grid)


def _integrate_numpy(
    rgb: np.ndarray,
    grid: np.ndarray,
    *,
    integration_steps: int,
    direction: float,
) -> np.ndarray:
    step = float(direction) / integration_steps
    current = rgb.copy()
    for _ in range(integration_steps):
        k1 = _velocity_numpy(current, grid)
        k2 = _velocity_numpy(current + 0.5 * step * k1, grid)
        k3 = _velocity_numpy(current + 0.5 * step * k2, grid)
        k4 = _velocity_numpy(current + step * k3, grid)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def _sample_grid_torch(rgb: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    axis_size = int(grid.shape[0])
    scaled = rgb * float(axis_size - 1)
    lower = torch.floor(scaled).to(dtype=torch.int64)
    lower = torch.minimum(
        lower,
        torch.full_like(lower, axis_size - 2),
    )
    fraction = scaled - lower.to(dtype=rgb.dtype)
    result = torch.zeros_like(rgb)
    for red in (0, 1):
        wr = fraction[:, 0] if red else 1.0 - fraction[:, 0]
        ir = lower[:, 0] + red
        for green in (0, 1):
            wg = fraction[:, 1] if green else 1.0 - fraction[:, 1]
            ig = lower[:, 1] + green
            for blue in (0, 1):
                wb = fraction[:, 2] if blue else 1.0 - fraction[:, 2]
                ib = lower[:, 2] + blue
                result = result + (wr * wg * wb)[:, None] * grid[ir, ig, ib]
    return result


def _velocity_torch(rgb: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return rgb * (1.0 - rgb) * _sample_grid_torch(rgb, grid)


def _integrate_torch(
    rgb: torch.Tensor,
    grid: torch.Tensor,
    *,
    integration_steps: int,
) -> torch.Tensor:
    step = 1.0 / integration_steps
    current = rgb
    for _ in range(integration_steps):
        k1 = _velocity_torch(current, grid)
        k2 = _velocity_torch(current + 0.5 * step * k1, grid)
        k3 = _velocity_torch(current + 0.5 * step * k2, grid)
        k4 = _velocity_torch(current + step * k3, grid)
        current = current + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    return current


@dataclass(frozen=True)
class CubeDiffeomorphicColourFlow:
    """Time-one flow of a stationary boundary-preserving RGB velocity field."""

    velocity_grid: np.ndarray
    integration_steps: int = 24
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        grid = _validate_velocity_grid(self.velocity_grid).copy()
        if self.integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        grid.setflags(write=False)
        object.__setattr__(self, "velocity_grid", grid)

    @classmethod
    def identity(
        cls, *, axis_size: int, integration_steps: int = 24
    ) -> "CubeDiffeomorphicColourFlow":
        if axis_size < 2:
            raise ValueError("axis_size must be at least two")
        return cls(
            np.zeros((axis_size, axis_size, axis_size, 3), dtype=np.float64),
            integration_steps=integration_steps,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        result = _integrate_numpy(
            values.reshape(-1, 3),
            self.velocity_grid,
            integration_steps=self.integration_steps,
            direction=1.0,
        )
        tolerance = 1e-12
        if (
            not np.all(np.isfinite(result))
            or np.any(result < -tolerance)
            or np.any(result > 1.0 + tolerance)
        ):
            raise RuntimeError("numerical colour flow escaped the RGB cube")
        return result.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        result = _integrate_numpy(
            values.reshape(-1, 3),
            self.velocity_grid,
            integration_steps=self.integration_steps,
            direction=-1.0,
        )
        tolerance = 1e-12
        if (
            not np.all(np.isfinite(result))
            or np.any(result < -tolerance)
            or np.any(result > 1.0 + tolerance)
        ):
            raise RuntimeError("numerical inverse colour flow escaped the RGB cube")
        return result.reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CUBE_DIFFEO_FLOW_SCHEMA,
            "working_space": self.working_space,
            "integration_steps": self.integration_steps,
            "velocity_grid": self.velocity_grid.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CubeDiffeomorphicColourFlow":
        if payload.get("schema") != CUBE_DIFFEO_FLOW_SCHEMA:
            raise ValueError("unsupported cube diffeomorphic colour-flow schema")
        return cls(
            velocity_grid=np.asarray(payload["velocity_grid"], dtype=np.float64),
            integration_steps=int(payload["integration_steps"]),
            working_space=str(payload["working_space"]),
        )


def _smoothness_loss(grid: torch.Tensor) -> torch.Tensor:
    losses = []
    for axis in range(3):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        losses.append(torch.mean((grid[tuple(right)] - grid[tuple(left)]) ** 2))
    return sum(losses)


def fit_cube_diffeomorphic_colour_flow(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    axis_size: int,
    integration_steps: int,
    maximum_absolute_coefficient: float,
    seed: int,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    velocity_smoothness_l2: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> CubeDiffeomorphicColourFlow:
    """Fit one explicit stationary velocity grid with deterministic CPU Adam."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    if axis_size < 2 or integration_steps < 1:
        raise ValueError("axis_size and integration_steps must be positive")
    if (
        not np.isfinite(maximum_absolute_coefficient)
        or maximum_absolute_coefficient <= 0.0
        or steps < 1
        or learning_rate <= 0.0
        or coefficient_l2 < 0.0
        or velocity_smoothness_l2 < 0.0
        or gradient_clip_norm <= 0.0
        or thread_count < 1
    ):
        raise ValueError("invalid fit settings")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(thread_count))
    source = torch.from_numpy(values.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(targets.reshape(-1, 3).copy())
    grid = torch.zeros(
        (axis_size, axis_size, axis_size, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([grid], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _integrate_torch(
            source,
            grid,
            integration_steps=integration_steps,
        )
        loss = torch.mean((prediction - target_tensor) ** 2)
        if coefficient_l2:
            loss = loss + coefficient_l2 * torch.mean(grid**2)
        if velocity_smoothness_l2:
            loss = loss + velocity_smoothness_l2 * _smoothness_loss(grid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([grid], float(gradient_clip_norm))
        optimizer.step()
        with torch.no_grad():
            grid.clamp_(
                -float(maximum_absolute_coefficient),
                float(maximum_absolute_coefficient),
            )
    return CubeDiffeomorphicColourFlow(
        velocity_grid=grid.detach().cpu().numpy(),
        integration_steps=integration_steps,
    )


def finite_difference_jacobians(
    operator: CubeDiffeomorphicColourFlow,
    points: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    values = _validate_rgb(points)
    if (
        values.ndim != 2
        or not np.isfinite(step)
        or step <= 0.0
        or np.any(values < step)
        or np.any(values > 1.0 - step)
    ):
        raise ValueError("points must be interior RGB rows for the given step")
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (operator.apply(values + offset) - operator.apply(values - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


__all__ = [
    "CUBE_DIFFEO_FLOW_SCHEMA",
    "CubeDiffeomorphicColourFlow",
    "finite_difference_jacobians",
    "fit_cube_diffeomorphic_colour_flow",
]
