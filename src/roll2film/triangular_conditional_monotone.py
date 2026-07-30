"""Cube-preserving triangular conditional monotone colour operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.roll2film.positive_film_fitting import PositiveFilmFitLoss


SCHEMA = "roll2film.triangular_conditional_monotone.v1"
_CHANNELS = ("R", "G", "B")


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


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _validate_order(order: tuple[int, int, int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in order)
    if sorted(result) != [0, 1, 2]:
        raise ValueError("channel order must be a permutation of RGB")
    return result


@dataclass(frozen=True)
class TriangularConditionalMonotoneOperator:
    """Triangular map with context-conditioned positive monotone curves."""

    channel_order: tuple[int, int, int]
    segment_count: int
    learned_mixture: float
    stage_parameters: tuple[np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self) -> None:
        order = _validate_order(self.channel_order)
        segments = int(self.segment_count)
        mixture = float(self.learned_mixture)
        if segments < 3 or not 0.0 < mixture < 1.0:
            raise ValueError("invalid conditional monotone curve contract")
        parameters = []
        for stage, values in enumerate(self.stage_parameters):
            array = np.asarray(values, dtype=np.float64)
            expected = (stage + 1, segments - 1)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError("conditional curve parameter shape drift")
            array = array.copy()
            array.setflags(write=False)
            parameters.append(array)
        object.__setattr__(self, "channel_order", order)
        object.__setattr__(self, "segment_count", segments)
        object.__setattr__(self, "learned_mixture", mixture)
        object.__setattr__(self, "stage_parameters", tuple(parameters))

    @property
    def order_name(self) -> str:
        return "".join(_CHANNELS[index] for index in self.channel_order)

    def _weights(
        self, stage: int, flat: np.ndarray
    ) -> np.ndarray:
        parameters = self.stage_parameters[stage]
        logits = np.zeros((len(flat), self.segment_count), dtype=np.float64)
        logits[:, 1:] = parameters[0]
        for previous in range(stage):
            channel = self.channel_order[previous]
            conditioner = 2.0 * flat[:, channel : channel + 1] - 1.0
            logits[:, 1:] += conditioner * parameters[previous + 1]
        probabilities = _softmax(logits)
        return (
            (1.0 - self.learned_mixture) / float(self.segment_count)
            + self.learned_mixture * probabilities
        )

    def _apply_and_slopes(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = _rgb(rgb)
        flat = values.reshape(-1, 3)
        output = np.empty_like(flat)
        slopes = np.empty_like(flat)
        for stage, channel in enumerate(self.channel_order):
            weights = self._weights(stage, flat)
            source_channel = flat[:, channel]
            scaled = source_channel * float(self.segment_count)
            indices = np.minimum(
                np.floor(scaled).astype(np.int64), self.segment_count - 1
            )
            fractions = scaled - indices
            fractions[source_channel == 1.0] = 1.0
            cumulative = np.concatenate(
                (
                    np.zeros((len(flat), 1), dtype=np.float64),
                    np.cumsum(weights, axis=1),
                ),
                axis=1,
            )
            rows = np.arange(len(flat))
            output[:, channel] = (
                cumulative[rows, indices]
                + fractions * weights[rows, indices]
            )
            slopes[:, channel] = (
                float(self.segment_count) * weights[rows, indices]
            )
        return output.reshape(values.shape), slopes.reshape(values.shape)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        output, _ = self._apply_and_slopes(rgb)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < -1e-12)
            or np.any(output > 1.0 + 1e-12)
        ):
            raise RuntimeError("conditional monotone operator escaped cube")
        return np.clip(output, 0.0, 1.0)

    def jacobian_determinants(self, rgb: np.ndarray) -> np.ndarray:
        values = _rgb(rgb)
        _, slopes = self._apply_and_slopes(values)
        return np.prod(slopes, axis=-1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "channel_order": list(self.channel_order),
            "order_name": self.order_name,
            "segment_count": self.segment_count,
            "learned_mixture": self.learned_mixture,
            "stage_parameters": [
                values.tolist() for values in self.stage_parameters
            ],
            "hard_output_clipping": False,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "TriangularConditionalMonotoneOperator":
        if (
            payload.get("schema") != SCHEMA
            or payload.get("hard_output_clipping") is not False
        ):
            raise ValueError("unsupported conditional monotone schema")
        operator = cls(
            channel_order=tuple(payload["channel_order"]),
            segment_count=int(payload["segment_count"]),
            learned_mixture=float(payload["learned_mixture"]),
            stage_parameters=tuple(
                np.asarray(values, dtype=np.float64)
                for values in payload["stage_parameters"]
            ),
        )
        if payload.get("order_name") != operator.order_name:
            raise ValueError("conditional monotone order identity drift")
        return operator


@dataclass(frozen=True)
class TriangularConditionalFitResult:
    operator: TriangularConditionalMonotoneOperator
    development_rgb_rmse: float
    function_evaluations: int
    restart_index: int
    converged: bool


def fit_triangular_conditional_monotone(
    source: np.ndarray,
    target: np.ndarray,
    *,
    channel_order: tuple[int, int, int],
    segment_count: int,
    learned_mixture: float,
    free_logit_bounds: tuple[float, float],
    conditioner_l2: float,
    restart_count: int,
    maximum_function_evaluations: int,
    function_tolerance: float,
    parameter_tolerance: float,
    gradient_tolerance: float,
    loss: PositiveFilmFitLoss,
    loss_scale: float,
    seed: int,
) -> TriangularConditionalFitResult:
    source_values = _rgb(source)
    target_values = _rgb(target)
    order = _validate_order(channel_order)
    if (
        source_values.ndim != 2
        or source_values.shape != target_values.shape
        or segment_count < 3
        or len(source_values) < 6 * segment_count
        or not 0.0 < learned_mixture < 1.0
        or not np.isfinite(conditioner_l2)
        or conditioner_l2 < 0.0
        or restart_count < 1
        or maximum_function_evaluations < 1
    ):
        raise ValueError("invalid conditional monotone fit contract")
    lower, upper = map(float, free_logit_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("invalid conditional monotone logit bounds")
    shapes = tuple(
        (stage + 1, segment_count - 1) for stage in range(3)
    )
    sizes = tuple(int(np.prod(shape)) for shape in shapes)
    parameter_count = sum(sizes)

    def operator(parameters: np.ndarray) -> TriangularConditionalMonotoneOperator:
        stages = []
        offset = 0
        for shape, size in zip(shapes, sizes, strict=True):
            stages.append(parameters[offset : offset + size].reshape(shape))
            offset += size
        return TriangularConditionalMonotoneOperator(
            channel_order=order,
            segment_count=segment_count,
            learned_mixture=learned_mixture,
            stage_parameters=tuple(stages),
        )

    coefficient_indices = []
    offset = 0
    for stage, size in enumerate(sizes):
        width = segment_count - 1
        coefficient_indices.extend(
            range(offset + width, offset + size)
        )
        offset += size
    coefficient_indices = np.asarray(coefficient_indices, dtype=np.int64)

    def residual(parameters: np.ndarray) -> np.ndarray:
        error = (
            operator(parameters).apply(source_values) - target_values
        ).reshape(-1)
        if conditioner_l2 == 0.0 or len(coefficient_indices) == 0:
            return error
        regularization = (
            np.sqrt(conditioner_l2)
            * parameters[coefficient_indices]
        )
        return np.concatenate((error, regularization))

    best = None
    best_operator = None
    best_restart = -1
    for restart in range(restart_count):
        initial = np.zeros(parameter_count, dtype=np.float64)
        if restart:
            rng = np.random.default_rng(seed + restart)
            initial += rng.normal(0.0, 0.1, parameter_count)
        initial = np.clip(initial, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            residual,
            initial,
            bounds=(
                np.full(parameter_count, lower),
                np.full(parameter_count, upper),
            ),
            method="trf",
            max_nfev=maximum_function_evaluations,
            ftol=function_tolerance,
            xtol=parameter_tolerance,
            gtol=gradient_tolerance,
            x_scale="jac",
            loss=loss,
            f_scale=loss_scale,
        )
        candidate = operator(result.x)
        if best is None or result.cost < best.cost:
            best = result
            best_operator = candidate
            best_restart = restart
    if best is None or best_operator is None:
        raise RuntimeError("conditional monotone fit produced no result")
    error = best_operator.apply(source_values) - target_values
    return TriangularConditionalFitResult(
        operator=best_operator,
        development_rgb_rmse=float(np.sqrt(np.mean(np.square(error)))),
        function_evaluations=int(best.nfev),
        restart_index=best_restart,
        converged=bool(best.success),
    )


__all__ = [
    "TriangularConditionalFitResult",
    "TriangularConditionalMonotoneOperator",
    "fit_triangular_conditional_monotone",
]
