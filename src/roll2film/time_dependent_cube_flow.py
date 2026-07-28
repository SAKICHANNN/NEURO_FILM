"""Explicit cube-preserving colour flows with a bounded time-varying velocity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .cube_diffeomorphic_flow import (
    _sample_grid_numpy,
    _sample_grid_torch,
    _smoothness_loss,
    _validate_rgb,
)


TIME_DEPENDENT_CUBE_FLOW_SCHEMA = (
    "roll2film.time_dependent_cube_colour_flow.v1"
)
_TEMPORAL_CONTROL_COUNT = 3


def _validate_control_grids(grids: np.ndarray) -> np.ndarray:
    values = np.asarray(grids, dtype=np.float64)
    if (
        values.ndim != 5
        or values.shape[0] != _TEMPORAL_CONTROL_COUNT
        or values.shape[1] != values.shape[2]
        or values.shape[2] != values.shape[3]
        or values.shape[4] != 3
        or values.shape[1] < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "control_grids must have finite shape (3, K, K, K, 3), K>=2"
        )
    return values


def _bernstein_weights_numpy(time: float) -> np.ndarray:
    if not np.isfinite(time) or time < 0.0 or time > 1.0:
        raise ValueError("time must be finite and in [0, 1]")
    one_minus = 1.0 - float(time)
    return np.asarray(
        [one_minus * one_minus, 2.0 * time * one_minus, time * time],
        dtype=np.float64,
    )


def _velocity_numpy(
    rgb: np.ndarray,
    time: float,
    control_grids: np.ndarray,
) -> np.ndarray:
    weights = _bernstein_weights_numpy(time)
    field = np.zeros_like(rgb)
    for index in range(_TEMPORAL_CONTROL_COUNT):
        field += weights[index] * _sample_grid_numpy(
            rgb, control_grids[index]
        )
    return rgb * (1.0 - rgb) * field


def _integrate_numpy(
    rgb: np.ndarray,
    control_grids: np.ndarray,
    *,
    integration_steps: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    step = (float(end_time) - float(start_time)) / integration_steps
    current = rgb.copy()
    for index in range(integration_steps):
        fraction = index / integration_steps
        next_fraction = (index + 1) / integration_steps
        time = start_time + (end_time - start_time) * fraction
        next_time = start_time + (end_time - start_time) * next_fraction
        midpoint = 0.5 * (time + next_time)
        k1 = _velocity_numpy(current, time, control_grids)
        k2 = _velocity_numpy(
            current + 0.5 * step * k1, midpoint, control_grids
        )
        k3 = _velocity_numpy(
            current + 0.5 * step * k2, midpoint, control_grids
        )
        k4 = _velocity_numpy(
            current + step * k3, next_time, control_grids
        )
        current = current + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    return current


def _bernstein_weights_torch(
    time: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    one_minus = 1.0 - float(time)
    return torch.tensor(
        [one_minus * one_minus, 2.0 * time * one_minus, time * time],
        dtype=dtype,
        device=device,
    )


def _velocity_torch(
    rgb: torch.Tensor,
    time: float,
    control_grids: torch.Tensor,
) -> torch.Tensor:
    weights = _bernstein_weights_torch(
        time,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    field = torch.zeros_like(rgb)
    for index in range(_TEMPORAL_CONTROL_COUNT):
        field = field + weights[index] * _sample_grid_torch(
            rgb, control_grids[index]
        )
    return rgb * (1.0 - rgb) * field


def _integrate_torch(
    rgb: torch.Tensor,
    control_grids: torch.Tensor,
    *,
    integration_steps: int,
) -> torch.Tensor:
    step = 1.0 / integration_steps
    current = rgb
    for index in range(integration_steps):
        time = index / integration_steps
        next_time = (index + 1) / integration_steps
        midpoint = 0.5 * (time + next_time)
        k1 = _velocity_torch(current, time, control_grids)
        k2 = _velocity_torch(
            current + 0.5 * step * k1, midpoint, control_grids
        )
        k3 = _velocity_torch(
            current + 0.5 * step * k2, midpoint, control_grids
        )
        k4 = _velocity_torch(
            current + step * k3, next_time, control_grids
        )
        current = current + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    return current


@dataclass(frozen=True)
class TimeDependentCubeColourFlow:
    """Time-one flow of a quadratic-time boundary-preserving RGB field."""

    control_grids: np.ndarray
    integration_steps: int = 24
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        grids = _validate_control_grids(self.control_grids).copy()
        if self.integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        grids.setflags(write=False)
        object.__setattr__(self, "control_grids", grids)

    @classmethod
    def identity(
        cls, *, axis_size: int, integration_steps: int = 24
    ) -> "TimeDependentCubeColourFlow":
        if axis_size < 2:
            raise ValueError("axis_size must be at least two")
        return cls(
            np.zeros(
                (_TEMPORAL_CONTROL_COUNT, axis_size, axis_size, axis_size, 3),
                dtype=np.float64,
            ),
            integration_steps=integration_steps,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        result = _integrate_numpy(
            values.reshape(-1, 3),
            self.control_grids,
            integration_steps=self.integration_steps,
            start_time=0.0,
            end_time=1.0,
        )
        self._validate_result(result, label="colour flow")
        return result.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        result = _integrate_numpy(
            values.reshape(-1, 3),
            self.control_grids,
            integration_steps=self.integration_steps,
            start_time=1.0,
            end_time=0.0,
        )
        self._validate_result(result, label="inverse colour flow")
        return result.reshape(shape)

    @staticmethod
    def _validate_result(result: np.ndarray, *, label: str) -> None:
        tolerance = 1e-12
        if (
            not np.all(np.isfinite(result))
            or np.any(result < -tolerance)
            or np.any(result > 1.0 + tolerance)
        ):
            raise RuntimeError(f"numerical {label} escaped the RGB cube")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TIME_DEPENDENT_CUBE_FLOW_SCHEMA,
            "working_space": self.working_space,
            "integration_steps": self.integration_steps,
            "temporal_basis": "quadratic_bernstein",
            "control_grids": self.control_grids.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "TimeDependentCubeColourFlow":
        if payload.get("schema") != TIME_DEPENDENT_CUBE_FLOW_SCHEMA:
            raise ValueError("unsupported time-dependent colour-flow schema")
        if payload.get("temporal_basis") != "quadratic_bernstein":
            raise ValueError("v1 requires quadratic_bernstein temporal basis")
        return cls(
            control_grids=np.asarray(
                payload["control_grids"], dtype=np.float64
            ),
            integration_steps=int(payload["integration_steps"]),
            working_space=str(payload["working_space"]),
        )


def fit_time_dependent_cube_colour_flow(
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
    spatial_smoothness_l2: float,
    temporal_smoothness_l2: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> TimeDependentCubeColourFlow:
    """Fit one explicit non-autonomous velocity field with deterministic Adam."""

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
        or spatial_smoothness_l2 < 0.0
        or temporal_smoothness_l2 < 0.0
        or gradient_clip_norm <= 0.0
        or thread_count < 1
    ):
        raise ValueError("invalid fit settings")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(thread_count))
    source = torch.from_numpy(values.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(targets.reshape(-1, 3).copy())
    grids = torch.zeros(
        (
            _TEMPORAL_CONTROL_COUNT,
            axis_size,
            axis_size,
            axis_size,
            3,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([grids], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _integrate_torch(
            source,
            grids,
            integration_steps=integration_steps,
        )
        loss = torch.mean((prediction - target_tensor) ** 2)
        if coefficient_l2:
            loss = loss + coefficient_l2 * torch.mean(grids**2)
        if spatial_smoothness_l2:
            spatial = sum(
                _smoothness_loss(grids[index])
                for index in range(_TEMPORAL_CONTROL_COUNT)
            ) / _TEMPORAL_CONTROL_COUNT
            loss = loss + spatial_smoothness_l2 * spatial
        if temporal_smoothness_l2:
            temporal = 0.5 * (
                torch.mean((grids[1] - grids[0]) ** 2)
                + torch.mean((grids[2] - grids[1]) ** 2)
            )
            loss = loss + temporal_smoothness_l2 * temporal
        loss.backward()
        torch.nn.utils.clip_grad_norm_([grids], float(gradient_clip_norm))
        optimizer.step()
        with torch.no_grad():
            grids.clamp_(
                -float(maximum_absolute_coefficient),
                float(maximum_absolute_coefficient),
            )
    return TimeDependentCubeColourFlow(
        control_grids=grids.detach().cpu().numpy(),
        integration_steps=integration_steps,
    )


__all__ = [
    "TIME_DEPENDENT_CUBE_FLOW_SCHEMA",
    "TimeDependentCubeColourFlow",
    "fit_time_dependent_cube_colour_flow",
]
