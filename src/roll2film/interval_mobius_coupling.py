"""Bounded explicit interval-Möbius colour-coupling operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


INTERVAL_MOBIUS_COUPLING_SCHEMA = "roll2film.interval_mobius_coupling.v1"
_FEATURE_COUNT = 6
_HEAD_COUNT = 3


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


def _conditioner_features_numpy(condition: np.ndarray) -> np.ndarray:
    centered = condition - 0.5
    left = centered[:, 0]
    right = centered[:, 1]
    return np.column_stack(
        (
            np.ones(len(condition), dtype=np.float64),
            left,
            right,
            left * right,
            left * left,
            right * right,
        )
    )


def _conditioner_features_torch(condition: torch.Tensor) -> torch.Tensor:
    centered = condition - 0.5
    left = centered[:, 0]
    right = centered[:, 1]
    return torch.stack(
        (
            torch.ones_like(left),
            left,
            right,
            left * right,
            left * left,
            right * right,
        ),
        dim=1,
    )


def _mobius_numpy(channel: np.ndarray, shift: np.ndarray) -> np.ndarray:
    odds = np.exp(shift)
    return channel * odds / (1.0 - channel + channel * odds)


def _stage_parameters_numpy(
    features: np.ndarray,
    coefficients: np.ndarray,
    *,
    maximum_lower_lift: float,
    maximum_upper_compression: float,
    maximum_log_odds_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = features @ coefficients.T
    lower = maximum_lower_lift * np.tanh(np.maximum(raw[:, 0], 0.0))
    upper = 1.0 - maximum_upper_compression * np.tanh(
        np.maximum(raw[:, 1], 0.0)
    )
    shift = maximum_log_odds_shift * np.tanh(raw[:, 2])
    return lower, upper, shift


@dataclass(frozen=True)
class IntervalMobiusCouplingOperator:
    """Composition of analytic bounded single-channel interval couplings."""

    stage_channels: tuple[int, ...]
    coefficients: np.ndarray
    maximum_lower_lift: float
    maximum_upper_compression: float
    maximum_log_odds_shift: float
    working_space: str = "linear_srgb_d65"

    def __post_init__(self) -> None:
        channels = tuple(int(value) for value in self.stage_channels)
        if not channels or any(value not in (0, 1, 2) for value in channels):
            raise ValueError("stage_channels must contain only 0, 1 and 2")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        expected = (len(channels), _HEAD_COUNT, _FEATURE_COUNT)
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"coefficients must have shape {expected}")
        limits = (
            self.maximum_lower_lift,
            self.maximum_upper_compression,
            self.maximum_log_odds_shift,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("all transform limits must be finite and positive")
        if self.maximum_lower_lift + self.maximum_upper_compression >= 1.0:
            raise ValueError("endpoint limits must leave positive interval width")
        if self.working_space != "linear_srgb_d65":
            raise ValueError("v1 requires linear_srgb_d65")
        coefficients = coefficients.copy()
        coefficients.setflags(write=False)
        object.__setattr__(self, "stage_channels", channels)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def identity(
        cls,
        *,
        stage_channels: Sequence[int],
        maximum_lower_lift: float,
        maximum_upper_compression: float,
        maximum_log_odds_shift: float,
    ) -> "IntervalMobiusCouplingOperator":
        channels = tuple(int(value) for value in stage_channels)
        return cls(
            stage_channels=channels,
            coefficients=np.zeros(
                (len(channels), _HEAD_COUNT, _FEATURE_COUNT),
                dtype=np.float64,
            ),
            maximum_lower_lift=maximum_lower_lift,
            maximum_upper_compression=maximum_upper_compression,
            maximum_log_odds_shift=maximum_log_odds_shift,
        )

    def _stage_parameters(
        self,
        rgb: np.ndarray,
        stage_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        channel = self.stage_channels[stage_index]
        left, right = _condition_channels(channel)
        condition = np.column_stack((rgb[:, left], rgb[:, right]))
        return _stage_parameters_numpy(
            _conditioner_features_numpy(condition),
            self.coefficients[stage_index],
            maximum_lower_lift=self.maximum_lower_lift,
            maximum_upper_compression=self.maximum_upper_compression,
            maximum_log_odds_shift=self.maximum_log_odds_shift,
        )

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index, channel_index in enumerate(self.stage_channels):
            lower, upper, shift = self._stage_parameters(current, stage_index)
            unit = _mobius_numpy(current[:, channel_index], shift)
            current[:, channel_index] = lower + (upper - lower) * unit
        if np.any(current < 0.0) or np.any(current > 1.0):
            raise RuntimeError("interval coupling escaped the RGB cube")
        return current.reshape(shape)

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        shape = values.shape
        current = values.reshape(-1, 3).copy()
        for stage_index in range(len(self.stage_channels) - 1, -1, -1):
            channel_index = self.stage_channels[stage_index]
            lower, upper, shift = self._stage_parameters(current, stage_index)
            unit = (current[:, channel_index] - lower) / (upper - lower)
            tolerance = 1e-10
            if np.any(unit < -tolerance) or np.any(unit > 1.0 + tolerance):
                raise ValueError("rgb lies outside the operator image")
            current[:, channel_index] = _mobius_numpy(unit, -shift)
        tolerance = 1e-10
        if np.any(current < -tolerance) or np.any(current > 1.0 + tolerance):
            raise RuntimeError("analytic coupling inverse escaped the RGB cube")
        return current.reshape(shape)

    def minimum_stage_derivative(self, rgb: np.ndarray) -> float:
        values = _validate_rgb(rgb)
        current = values.reshape(-1, 3).copy()
        minimum = np.inf
        for stage_index, channel_index in enumerate(self.stage_channels):
            lower, upper, shift = self._stage_parameters(current, stage_index)
            channel = current[:, channel_index]
            odds = np.exp(shift)
            denominator = 1.0 - channel + channel * odds
            derivative = (upper - lower) * odds / (denominator * denominator)
            minimum = min(minimum, float(np.min(derivative)))
            unit = channel * odds / denominator
            current[:, channel_index] = lower + (upper - lower) * unit
        return float(minimum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERVAL_MOBIUS_COUPLING_SCHEMA,
            "working_space": self.working_space,
            "stage_channels": list(self.stage_channels),
            "maximum_lower_lift": self.maximum_lower_lift,
            "maximum_upper_compression": self.maximum_upper_compression,
            "maximum_log_odds_shift": self.maximum_log_odds_shift,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntervalMobiusCouplingOperator":
        if payload.get("schema") != INTERVAL_MOBIUS_COUPLING_SCHEMA:
            raise ValueError("unsupported interval-Mobius coupling schema")
        return cls(
            stage_channels=tuple(int(value) for value in payload["stage_channels"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            maximum_lower_lift=float(payload["maximum_lower_lift"]),
            maximum_upper_compression=float(payload["maximum_upper_compression"]),
            maximum_log_odds_shift=float(payload["maximum_log_odds_shift"]),
            working_space=str(payload["working_space"]),
        )


def _torch_apply(
    rgb: torch.Tensor,
    stage_channels: tuple[int, ...],
    coefficients: torch.Tensor,
    *,
    maximum_lower_lift: float,
    maximum_upper_compression: float,
    maximum_log_odds_shift: float,
) -> torch.Tensor:
    current = rgb
    for stage_index, channel_index in enumerate(stage_channels):
        left, right = _condition_channels(channel_index)
        condition = torch.stack((current[:, left], current[:, right]), dim=1)
        features = _conditioner_features_torch(condition)
        raw = features @ coefficients[stage_index].T
        lower = maximum_lower_lift * torch.tanh(torch.clamp_min(raw[:, 0], 0.0))
        upper = 1.0 - maximum_upper_compression * torch.tanh(
            torch.clamp_min(raw[:, 1], 0.0)
        )
        shift = maximum_log_odds_shift * torch.tanh(raw[:, 2])
        odds = torch.exp(shift)
        channel = current[:, channel_index]
        unit = channel * odds / (1.0 - channel + channel * odds)
        updated = lower + (upper - lower) * unit
        current = torch.stack(
            [
                updated if index == channel_index else current[:, index]
                for index in range(3)
            ],
            dim=1,
        )
    return current


def fit_interval_mobius_coupling(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    stage_channels: Sequence[int],
    maximum_lower_lift: float,
    maximum_upper_compression: float,
    maximum_log_odds_shift: float,
    maximum_absolute_coefficient: float,
    endpoint_head_bias_initialization: float,
    log_odds_head_initialization: float,
    seed: int,
    steps: int,
    learning_rate: float,
    coefficient_l2: float,
    gradient_clip_norm: float,
    thread_count: int,
) -> IntervalMobiusCouplingOperator:
    """Fit explicit coupling coefficients with deterministic CPU float64 Adam."""

    values = _validate_rgb(rgb)
    targets = _validate_rgb(target)
    if targets.shape != values.shape:
        raise ValueError("target must have the same shape as rgb")
    channels = tuple(int(value) for value in stage_channels)
    if not channels or any(value not in (0, 1, 2) for value in channels):
        raise ValueError("invalid stage_channels")
    numeric_positive = (
        maximum_lower_lift,
        maximum_upper_compression,
        maximum_log_odds_shift,
        maximum_absolute_coefficient,
        learning_rate,
        gradient_clip_norm,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in numeric_positive):
        raise ValueError("positive fit settings must be finite and positive")
    if maximum_lower_lift + maximum_upper_compression >= 1.0:
        raise ValueError("endpoint limits must leave positive interval width")
    if (
        not np.isfinite(endpoint_head_bias_initialization)
        or endpoint_head_bias_initialization <= 0.0
        or not np.isfinite(log_odds_head_initialization)
    ):
        raise ValueError("invalid coefficient initialization")
    if steps <= 0 or coefficient_l2 < 0.0 or thread_count <= 0:
        raise ValueError("invalid optimizer settings")

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(thread_count))
    source_tensor = torch.from_numpy(values.reshape(-1, 3).copy())
    target_tensor = torch.from_numpy(targets.reshape(-1, 3).copy())
    initial = np.zeros(
        (len(channels), _HEAD_COUNT, _FEATURE_COUNT),
        dtype=np.float64,
    )
    initial[:, 0, 0] = endpoint_head_bias_initialization
    initial[:, 1, 0] = endpoint_head_bias_initialization
    initial[:, 2, 0] = log_odds_head_initialization
    coefficients = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=float(learning_rate))
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _torch_apply(
            source_tensor,
            channels,
            coefficients,
            maximum_lower_lift=maximum_lower_lift,
            maximum_upper_compression=maximum_upper_compression,
            maximum_log_odds_shift=maximum_log_odds_shift,
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
    return IntervalMobiusCouplingOperator(
        stage_channels=channels,
        coefficients=coefficients.detach().cpu().numpy(),
        maximum_lower_lift=maximum_lower_lift,
        maximum_upper_compression=maximum_upper_compression,
        maximum_log_odds_shift=maximum_log_odds_shift,
    )


def finite_difference_jacobians(
    operator: IntervalMobiusCouplingOperator,
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
    "INTERVAL_MOBIUS_COUPLING_SCHEMA",
    "IntervalMobiusCouplingOperator",
    "finite_difference_jacobians",
    "fit_interval_mobius_coupling",
]
