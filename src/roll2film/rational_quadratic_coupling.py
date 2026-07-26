"""Explicit bounded rational-quadratic colour-coupling operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.roll2film.monotone_coupling import regular_conditioner_centers


RATIONAL_QUADRATIC_COUPLING_SCHEMA = (
    "roll2film.rational_quadratic_coupling.v1"
)


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


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _inverse_softplus(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("inverse softplus input must be finite and positive")
    return float(np.log(np.expm1(value)))


def _spline_parameters_numpy(
    raw: np.ndarray,
    *,
    bin_count: int,
    minimum_bin_width: float,
    minimum_bin_height: float,
    minimum_derivative: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width_logits = raw[:, :bin_count]
    height_logits = raw[:, bin_count : 2 * bin_count]
    derivative_offsets = raw[:, 2 * bin_count :]
    widths = minimum_bin_width + (
        1.0 - minimum_bin_width * bin_count
    ) * _softmax_numpy(width_logits)
    heights = minimum_bin_height + (
        1.0 - minimum_bin_height * bin_count
    ) * _softmax_numpy(height_logits)
    derivative_base = _inverse_softplus(1.0 - minimum_derivative)
    internal = minimum_derivative + np.logaddexp(
        0.0, derivative_base + derivative_offsets
    )
    derivatives = np.column_stack(
        (
            np.ones(len(raw), dtype=np.float64),
            internal,
            np.ones(len(raw), dtype=np.float64),
        )
    )
    return widths, heights, derivatives


def _spline_parameters_torch(
    raw: torch.Tensor,
    *,
    bin_count: int,
    minimum_bin_width: float,
    minimum_bin_height: float,
    minimum_derivative: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    widths = minimum_bin_width + (
        1.0 - minimum_bin_width * bin_count
    ) * torch.softmax(raw[:, :bin_count], dim=1)
    heights = minimum_bin_height + (
        1.0 - minimum_bin_height * bin_count
    ) * torch.softmax(raw[:, bin_count : 2 * bin_count], dim=1)
    derivative_base = _inverse_softplus(1.0 - minimum_derivative)
    internal = minimum_derivative + torch.nn.functional.softplus(
        raw[:, 2 * bin_count :] + derivative_base
    )
    derivatives = torch.cat(
        (
            torch.ones(
                (len(raw), 1), dtype=raw.dtype, device=raw.device
            ),
            internal,
            torch.ones(
                (len(raw), 1), dtype=raw.dtype, device=raw.device
            ),
        ),
        dim=1,
    )
    return widths, heights, derivatives


def _cumulative_numpy(values: np.ndarray) -> np.ndarray:
    cumulative = np.column_stack(
        (np.zeros(len(values), dtype=np.float64), np.cumsum(values, axis=1))
    )
    cumulative[:, -1] = 1.0
    return cumulative


def _gather_numpy(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices[:, None], axis=1)[:, 0]


def _rational_quadratic_numpy(
    inputs: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    derivatives: np.ndarray,
    *,
    inverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative_widths = _cumulative_numpy(widths)
    cumulative_heights = _cumulative_numpy(heights)
    knots = cumulative_heights if inverse else cumulative_widths
    indices = np.sum(inputs[:, None] >= knots[:, 1:-1], axis=1)
    x_left = _gather_numpy(cumulative_widths, indices)
    y_left = _gather_numpy(cumulative_heights, indices)
    width = _gather_numpy(widths, indices)
    height = _gather_numpy(heights, indices)
    derivative_left = _gather_numpy(derivatives, indices)
    derivative_right = _gather_numpy(derivatives, indices + 1)
    slope = height / width

    if inverse:
        y_delta = inputs - y_left
        common = derivative_left + derivative_right - 2.0 * slope
        a = y_delta * common + height * (slope - derivative_left)
        b = height * derivative_left - y_delta * common
        c = -slope * y_delta
        discriminant = b * b - 4.0 * a * c
        if np.any(discriminant < -1e-12):
            raise RuntimeError("rational-quadratic inverse has negative discriminant")
        root = (2.0 * c) / (
            -b - np.sqrt(np.maximum(discriminant, 0.0))
        )
        theta = root
        outputs = x_left + theta * width
    else:
        theta = (inputs - x_left) / width
        theta_one_minus = theta * (1.0 - theta)
        numerator = height * (
            slope * theta * theta
            + derivative_left * theta_one_minus
        )
        denominator = slope + (
            derivative_right + derivative_left - 2.0 * slope
        ) * theta_one_minus
        outputs = y_left + numerator / denominator

    theta_one_minus = theta * (1.0 - theta)
    denominator = slope + (
        derivative_right + derivative_left - 2.0 * slope
    ) * theta_one_minus
    derivative_numerator = slope * slope * (
        derivative_right * theta * theta
        + 2.0 * slope * theta_one_minus
        + derivative_left * (1.0 - theta) ** 2
    )
    stage_derivative = derivative_numerator / (denominator * denominator)
    outputs = np.where(inputs == 0.0, 0.0, outputs)
    outputs = np.where(inputs == 1.0, 1.0, outputs)
    return outputs, stage_derivative


def _rational_quadratic_torch(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
) -> torch.Tensor:
    cumulative_widths = torch.cat(
        (
            torch.zeros(
                (len(widths), 1), dtype=widths.dtype, device=widths.device
            ),
            torch.cumsum(widths, dim=1),
        ),
        dim=1,
    )
    cumulative_heights = torch.cat(
        (
            torch.zeros(
                (len(heights), 1), dtype=heights.dtype, device=heights.device
            ),
            torch.cumsum(heights, dim=1),
        ),
        dim=1,
    )
    indices = torch.sum(
        inputs[:, None] >= cumulative_widths[:, 1:-1], dim=1
    ).to(torch.int64)

    def gather(values: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return torch.gather(values, 1, (indices + offset)[:, None])[:, 0]

    x_left = gather(cumulative_widths)
    y_left = gather(cumulative_heights)
    width = gather(widths)
    height = gather(heights)
    derivative_left = gather(derivatives)
    derivative_right = gather(derivatives, 1)
    slope = height / width
    theta = (inputs - x_left) / width
    theta_one_minus = theta * (1.0 - theta)
    numerator = height * (
        slope * theta * theta + derivative_left * theta_one_minus
    )
    denominator = slope + (
        derivative_right + derivative_left - 2.0 * slope
    ) * theta_one_minus
    return y_left + numerator / denominator


@dataclass(frozen=True)
class RationalQuadraticCouplingOperator:
    """Composition of finite monotonic rational-quadratic coupling stages."""

    stage_channels: tuple[int, ...]
    centers: np.ndarray
    sigma: float
    bin_count: int
    minimum_bin_width: float
    minimum_bin_height: float
    minimum_derivative: float
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
        if self.bin_count < 2:
            raise ValueError("bin_count must be at least two")
        for name, value in (
            ("minimum_bin_width", self.minimum_bin_width),
            ("minimum_bin_height", self.minimum_bin_height),
            ("minimum_derivative", self.minimum_derivative),
            ("sigma", self.sigma),
            ("epsilon", self.epsilon),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_bin_width * self.bin_count >= 1.0:
            raise ValueError("minimum bin widths exhaust the unit interval")
        if self.minimum_bin_height * self.bin_count >= 1.0:
            raise ValueError("minimum bin heights exhaust the unit interval")
        feature_count = 3 + len(centers)
        parameter_count = 3 * self.bin_count - 1
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(channels), feature_count, parameter_count)
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"coefficients must have shape {expected}")
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
        bin_count: int,
        minimum_bin_width: float,
        minimum_bin_height: float,
        minimum_derivative: float,
        epsilon: float = 1e-12,
    ) -> "RationalQuadraticCouplingOperator":
        channels = tuple(int(value) for value in stage_channels)
        centers = regular_conditioner_centers(axis_size)
        return cls(
            stage_channels=channels,
            centers=centers,
            sigma=sigma,
            bin_count=bin_count,
            minimum_bin_width=minimum_bin_width,
            minimum_bin_height=minimum_bin_height,
            minimum_derivative=minimum_derivative,
            epsilon=epsilon,
            coefficients=np.zeros(
                (len(channels), 3 + len(centers), 3 * bin_count - 1),
                dtype=np.float64,
            ),
        )

    def _stage_parameters(
        self, rgb: np.ndarray, stage_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        channel = self.stage_channels[stage_index]
        left, right = _condition_channels(channel)
        condition = np.column_stack((rgb[:, left], rgb[:, right]))
        features = _conditioner_features_numpy(
            condition,
            self.centers,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        raw = np.sum(
            features[:, :, None]
            * self.coefficients[stage_index][None, :, :],
            axis=1,
        )
        return _spline_parameters_numpy(
            raw,
            bin_count=self.bin_count,
            minimum_bin_width=self.minimum_bin_width,
            minimum_bin_height=self.minimum_bin_height,
            minimum_derivative=self.minimum_derivative,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index, channel in enumerate(self.stage_channels):
            widths, heights, derivatives = self._stage_parameters(
                current, stage_index
            )
            current[:, channel], _ = _rational_quadratic_numpy(
                current[:, channel],
                widths,
                heights,
                derivatives,
                inverse=False,
            )
        tolerance = 1e-12
        if np.any(current < -tolerance) or np.any(current > 1.0 + tolerance):
            raise RuntimeError("rational-quadratic coupling escaped the RGB cube")
        return current.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index in range(len(self.stage_channels) - 1, -1, -1):
            channel = self.stage_channels[stage_index]
            widths, heights, derivatives = self._stage_parameters(
                current, stage_index
            )
            current[:, channel], _ = _rational_quadratic_numpy(
                current[:, channel],
                widths,
                heights,
                derivatives,
                inverse=True,
            )
        tolerance = 1e-10
        if np.any(current < -tolerance) or np.any(current > 1.0 + tolerance):
            raise RuntimeError("rational-quadratic inverse escaped the RGB cube")
        return current.reshape(shape)

    def stage_statistics(self, rgb: np.ndarray) -> dict[str, float]:
        values = _validate_rgb(rgb)
        current = values.reshape(-1, 3).copy()
        minimum_width = np.inf
        minimum_height = np.inf
        minimum_stage_derivative = np.inf
        for stage_index, channel in enumerate(self.stage_channels):
            widths, heights, derivatives = self._stage_parameters(
                current, stage_index
            )
            output, stage_derivative = _rational_quadratic_numpy(
                current[:, channel],
                widths,
                heights,
                derivatives,
                inverse=False,
            )
            minimum_width = min(minimum_width, float(np.min(widths)))
            minimum_height = min(minimum_height, float(np.min(heights)))
            minimum_stage_derivative = min(
                minimum_stage_derivative, float(np.min(stage_derivative))
            )
            current[:, channel] = output
        return {
            "minimum_bin_width": float(minimum_width),
            "minimum_bin_height": float(minimum_height),
            "minimum_stage_derivative": float(minimum_stage_derivative),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RATIONAL_QUADRATIC_COUPLING_SCHEMA,
            "working_space": self.working_space,
            "stage_channels": list(self.stage_channels),
            "sigma": self.sigma,
            "bin_count": self.bin_count,
            "minimum_bin_width": self.minimum_bin_width,
            "minimum_bin_height": self.minimum_bin_height,
            "minimum_derivative": self.minimum_derivative,
            "epsilon": self.epsilon,
            "centers": self.centers.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "RationalQuadraticCouplingOperator":
        if payload.get("schema") != RATIONAL_QUADRATIC_COUPLING_SCHEMA:
            raise ValueError("unsupported rational-quadratic coupling schema")
        return cls(
            stage_channels=tuple(int(value) for value in payload["stage_channels"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            sigma=float(payload["sigma"]),
            bin_count=int(payload["bin_count"]),
            minimum_bin_width=float(payload["minimum_bin_width"]),
            minimum_bin_height=float(payload["minimum_bin_height"]),
            minimum_derivative=float(payload["minimum_derivative"]),
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
    bin_count: int,
    minimum_bin_width: float,
    minimum_bin_height: float,
    minimum_derivative: float,
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
        raw = features @ coefficients[stage_index]
        widths, heights, derivatives = _spline_parameters_torch(
            raw,
            bin_count=bin_count,
            minimum_bin_width=minimum_bin_width,
            minimum_bin_height=minimum_bin_height,
            minimum_derivative=minimum_derivative,
        )
        updated = _rational_quadratic_torch(
            current[:, channel], widths, heights, derivatives
        )
        current = torch.stack(
            [
                updated if index == channel else current[:, index]
                for index in range(3)
            ],
            dim=1,
        )
    return current


def fit_rational_quadratic_coupling(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    stage_channels: Sequence[int],
    axis_size: int,
    sigma: float,
    bin_count: int,
    minimum_bin_width: float,
    minimum_bin_height: float,
    minimum_derivative: float,
    epsilon: float,
    maximum_absolute_coefficient: float,
    seed: int,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> RationalQuadraticCouplingOperator:
    """Fit finite spline coefficients with deterministic CPU float64 Adam."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    identity = RationalQuadraticCouplingOperator.identity(
        stage_channels=stage_channels,
        axis_size=axis_size,
        sigma=sigma,
        bin_count=bin_count,
        minimum_bin_width=minimum_bin_width,
        minimum_bin_height=minimum_bin_height,
        minimum_derivative=minimum_derivative,
        epsilon=epsilon,
    )
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
    centers = torch.from_numpy(identity.centers.copy())
    coefficients = torch.zeros(
        identity.coefficients.shape, dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([coefficients], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _torch_apply(
            source_tensor,
            identity.stage_channels,
            centers,
            coefficients,
            sigma=sigma,
            epsilon=epsilon,
            bin_count=bin_count,
            minimum_bin_width=minimum_bin_width,
            minimum_bin_height=minimum_bin_height,
            minimum_derivative=minimum_derivative,
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
    return RationalQuadraticCouplingOperator(
        stage_channels=identity.stage_channels,
        centers=identity.centers,
        sigma=sigma,
        bin_count=bin_count,
        minimum_bin_width=minimum_bin_width,
        minimum_bin_height=minimum_bin_height,
        minimum_derivative=minimum_derivative,
        epsilon=epsilon,
        coefficients=coefficients.detach().cpu().numpy(),
    )


def finite_difference_jacobians(
    operator: RationalQuadraticCouplingOperator,
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
    "RATIONAL_QUADRATIC_COUPLING_SCHEMA",
    "RationalQuadraticCouplingOperator",
    "finite_difference_jacobians",
    "fit_rational_quadratic_coupling",
]
