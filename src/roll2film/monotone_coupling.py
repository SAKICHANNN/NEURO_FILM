"""Explicit bounded triangular monotone colour-coupling operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


MONOTONE_COUPLING_SCHEMA = "roll2film.triangular_monotone_coupling.v1"


def regular_conditioner_centers(axis_size: int) -> np.ndarray:
    if axis_size < 2:
        raise ValueError("axis_size must be at least two")
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


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


def _condition_channels(channel: int) -> tuple[int, int]:
    if channel not in (0, 1, 2):
        raise ValueError("channel must be 0, 1 or 2")
    return tuple(index for index in range(3) if index != channel)  # type: ignore[return-value]


def _conditioner_features_numpy(
    condition: np.ndarray,
    centers: np.ndarray,
    *,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    squared = np.sum(
        (condition[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    density = np.exp(-0.5 * squared / (sigma * sigma))
    weights = density / (np.sum(density, axis=1, keepdims=True) + epsilon)
    return np.column_stack(
        (np.ones(len(condition), dtype=np.float64), condition, weights)
    )


def _conditioner_features_torch(
    condition: torch.Tensor,
    centers: torch.Tensor,
    *,
    sigma: float,
    epsilon: float,
) -> torch.Tensor:
    squared = torch.sum(
        (condition[:, None, :] - centers[None, :, :]) ** 2,
        dim=2,
    )
    density = torch.exp(-0.5 * squared / (sigma * sigma))
    weights = density / (torch.sum(density, dim=1, keepdim=True) + epsilon)
    return torch.cat(
        (
            torch.ones(
                (len(condition), 1),
                dtype=condition.dtype,
                device=condition.device,
            ),
            condition,
            weights,
        ),
        dim=1,
    )


def _bounded_channel_forward(channel: np.ndarray, delta: np.ndarray) -> np.ndarray:
    positive = np.maximum(delta, 0.0)
    negative = np.minimum(delta, 0.0)
    return (
        channel
        + (1.0 - channel) * np.tanh(positive)
        + channel * np.tanh(negative)
    )


def _bounded_channel_inverse(channel: np.ndarray, delta: np.ndarray) -> np.ndarray:
    shift = np.tanh(delta)
    return np.where(
        delta >= 0.0,
        (channel - shift) / (1.0 - shift),
        channel / (1.0 + shift),
    )


@dataclass(frozen=True)
class TriangularMonotoneCouplingOperator:
    """Composition of single-channel bounded monotone coupling stages."""

    stage_channels: tuple[int, ...]
    centers: np.ndarray
    sigma: float
    coefficients: np.ndarray
    epsilon: float = 1e-12
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        channels = tuple(int(value) for value in self.stage_channels)
        if not channels or any(value not in (0, 1, 2) for value in channels):
            raise ValueError("stage_channels must contain only 0, 1 and 2")
        centers = np.asarray(self.centers, dtype=np.float64)
        if (
            centers.ndim != 2
            or centers.shape[1] != 2
            or len(centers) == 0
            or not np.all(np.isfinite(centers))
            or np.any(centers < 0.0)
            or np.any(centers > 1.0)
        ):
            raise ValueError("centers must be finite [0, 1] rows with shape (N, 2)")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(channels), 3 + len(centers))
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"coefficients must have shape {expected}")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        centers = centers.copy()
        coefficients = coefficients.copy()
        centers.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "stage_channels", channels)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def identity(
        cls,
        *,
        stage_channels: Sequence[int],
        axis_size: int,
        sigma: float,
        epsilon: float = 1e-12,
    ) -> "TriangularMonotoneCouplingOperator":
        centers = regular_conditioner_centers(axis_size)
        return cls(
            stage_channels=tuple(stage_channels),
            centers=centers,
            sigma=sigma,
            epsilon=epsilon,
            coefficients=np.zeros(
                (len(tuple(stage_channels)), 3 + len(centers)),
                dtype=np.float64,
            ),
        )

    def _stage_delta(self, rgb: np.ndarray, stage_index: int) -> np.ndarray:
        channel = self.stage_channels[stage_index]
        left, right = _condition_channels(channel)
        condition = np.column_stack((rgb[:, left], rgb[:, right]))
        features = _conditioner_features_numpy(
            condition,
            self.centers,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        return features @ self.coefficients[stage_index]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index, channel in enumerate(self.stage_channels):
            delta = self._stage_delta(current, stage_index)
            current[:, channel] = _bounded_channel_forward(
                current[:, channel], delta
            )
        return current.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index in range(len(self.stage_channels) - 1, -1, -1):
            channel = self.stage_channels[stage_index]
            delta = self._stage_delta(current, stage_index)
            current[:, channel] = _bounded_channel_inverse(
                current[:, channel], delta
            )
        tolerance = 1e-10
        if np.any(current < -tolerance) or np.any(current > 1.0 + tolerance):
            raise RuntimeError("analytic coupling inverse escaped the RGB cube")
        return current.reshape(shape)

    def minimum_stage_derivative(self, rgb: np.ndarray) -> float:
        values = _validate_rgb(rgb)
        current = values.reshape(-1, 3).copy()
        minimum = np.inf
        for stage_index, channel in enumerate(self.stage_channels):
            delta = self._stage_delta(current, stage_index)
            minimum = min(minimum, float(np.min(1.0 - np.abs(np.tanh(delta)))))
            current[:, channel] = _bounded_channel_forward(
                current[:, channel], delta
            )
        return float(minimum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MONOTONE_COUPLING_SCHEMA,
            "working_space": self.working_space,
            "stage_channels": list(self.stage_channels),
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "centers": self.centers.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TriangularMonotoneCouplingOperator":
        if payload.get("schema") != MONOTONE_COUPLING_SCHEMA:
            raise ValueError("unsupported triangular monotone coupling schema")
        return cls(
            stage_channels=tuple(int(value) for value in payload["stage_channels"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            epsilon=float(payload["epsilon"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            working_space=str(payload["working_space"]),
        )


def _torch_apply(
    rgb: torch.Tensor,
    stage_channels: tuple[int, ...],
    centers: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    sigma: float,
    epsilon: float,
) -> torch.Tensor:
    current = rgb
    for stage_index, channel in enumerate(stage_channels):
        left, right = _condition_channels(channel)
        condition = torch.stack((current[:, left], current[:, right]), dim=1)
        features = _conditioner_features_torch(
            condition,
            centers,
            sigma=sigma,
            epsilon=epsilon,
        )
        delta = features @ coefficients[stage_index]
        shift = torch.tanh(delta)
        updated = torch.where(
            delta >= 0.0,
            current[:, channel] + (1.0 - current[:, channel]) * shift,
            current[:, channel] + current[:, channel] * shift,
        )
        next_channels = [
            updated if index == channel else current[:, index]
            for index in range(3)
        ]
        current = torch.stack(next_channels, dim=1)
    return current


def fit_triangular_monotone_coupling(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    stage_channels: Sequence[int],
    axis_size: int,
    sigma: float,
    epsilon: float,
    maximum_absolute_coefficient: float,
    seed: int,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> TriangularMonotoneCouplingOperator:
    """Fit explicit coupling coefficients with deterministic CPU float64 Adam."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    channels = tuple(int(value) for value in stage_channels)
    if not channels or any(value not in (0, 1, 2) for value in channels):
        raise ValueError("invalid stage_channels")
    if not np.isfinite(maximum_absolute_coefficient) or (
        maximum_absolute_coefficient <= 0.0
    ):
        raise ValueError("maximum_absolute_coefficient must be positive")
    if steps <= 0 or learning_rate <= 0.0 or coefficient_l2 < 0.0:
        raise ValueError("invalid optimizer settings")
    if gradient_clip_norm <= 0.0 or thread_count <= 0:
        raise ValueError("invalid execution settings")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(thread_count))
    source_tensor = torch.from_numpy(values.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(targets.reshape(-1, 3).copy())
    centers_numpy = regular_conditioner_centers(axis_size)
    centers = torch.from_numpy(centers_numpy.copy())
    coefficients = torch.zeros(
        (len(channels), 3 + len(centers_numpy)),
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([coefficients], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _torch_apply(
            source_tensor,
            channels,
            centers,
            coefficients,
            sigma=sigma,
            epsilon=epsilon,
        )
        loss = torch.mean((prediction - target_tensor) ** 2)
        if coefficient_l2:
            loss = loss + coefficient_l2 * torch.mean(coefficients**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coefficients], gradient_clip_norm)
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(
                -maximum_absolute_coefficient,
                maximum_absolute_coefficient,
            )
    return TriangularMonotoneCouplingOperator(
        stage_channels=channels,
        centers=centers_numpy,
        sigma=sigma,
        epsilon=epsilon,
        coefficients=coefficients.detach().cpu().numpy(),
    )


def finite_difference_jacobians(
    operator: TriangularMonotoneCouplingOperator,
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
        raise ValueError("points must be finite interior RGB rows for the given step")
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
    "MONOTONE_COUPLING_SCHEMA",
    "TriangularMonotoneCouplingOperator",
    "finite_difference_jacobians",
    "fit_triangular_monotone_coupling",
    "regular_conditioner_centers",
]
