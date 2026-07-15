"""Invertible monotone splines and the Roll2Film L2 colour operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import AffineColorOperator, _validate_rgb


SPLINE_SCHEMA = "roll2film.rational_quadratic_spline.v1"
L2_OPERATOR_SCHEMA = "roll2film.affine_monotone_spline.v1"


@dataclass(frozen=True)
class RationalQuadraticSpline:
    """Strictly monotone scalar rational-quadratic spline with linear tails."""

    x_knots: np.ndarray
    y_knots: np.ndarray
    derivatives: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x_knots, dtype=np.float64)
        y = np.asarray(self.y_knots, dtype=np.float64)
        derivatives = np.asarray(self.derivatives, dtype=np.float64)
        if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or derivatives.shape != x.shape:
            raise ValueError("spline knots and derivatives must be equal one-dimensional arrays")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("spline knots must be finite")
        if not np.all(np.isfinite(derivatives)) or np.any(derivatives <= 0.0):
            raise ValueError("spline derivatives must be finite and strictly positive")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("spline knots must be strictly increasing")
        object.__setattr__(self, "x_knots", x)
        object.__setattr__(self, "y_knots", y)
        object.__setattr__(self, "derivatives", derivatives)

    @classmethod
    def identity(cls, minimum: float = 0.0, maximum: float = 1.0) -> "RationalQuadraticSpline":
        if maximum <= minimum:
            raise ValueError("identity spline maximum must exceed minimum")
        knots = np.array([minimum, maximum], dtype=np.float64)
        return cls(knots, knots.copy(), np.ones(2, dtype=np.float64))

    @classmethod
    def from_knots(cls, x_knots: np.ndarray, y_knots: np.ndarray) -> "RationalQuadraticSpline":
        x = np.asarray(x_knots, dtype=np.float64)
        y = np.asarray(y_knots, dtype=np.float64)
        if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
            raise ValueError("x_knots and y_knots must have the same one-dimensional shape")
        widths = np.diff(x)
        heights = np.diff(y)
        if np.any(widths <= 0.0) or np.any(heights <= 0.0):
            raise ValueError("spline knots must be strictly increasing")
        slopes = heights / widths
        derivatives = np.empty_like(x)
        derivatives[0] = slopes[0]
        derivatives[-1] = slopes[-1]
        if len(x) > 2:
            left = widths[:-1]
            right = widths[1:]
            derivatives[1:-1] = (left + right) / (
                left / slopes[:-1] + right / slopes[1:]
            )
        return cls(x, y, derivatives)

    def apply(self, values: np.ndarray) -> np.ndarray:
        inputs = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(inputs)):
            raise ValueError("spline inputs must be finite")
        if np.array_equal(self.x_knots, self.y_knots) and np.array_equal(
            self.derivatives, np.ones_like(self.derivatives)
        ):
            return inputs.copy()
        outputs = np.empty_like(inputs)
        below = inputs < self.x_knots[0]
        above = inputs > self.x_knots[-1]
        inside = ~(below | above)
        outputs[below] = self.y_knots[0] + self.derivatives[0] * (
            inputs[below] - self.x_knots[0]
        )
        outputs[above] = self.y_knots[-1] + self.derivatives[-1] * (
            inputs[above] - self.x_knots[-1]
        )
        if np.any(inside):
            outputs[inside] = self._forward_inside(inputs[inside])[0]
        return outputs

    def inverse(self, values: np.ndarray) -> np.ndarray:
        outputs = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(outputs)):
            raise ValueError("spline outputs must be finite")
        if np.array_equal(self.x_knots, self.y_knots) and np.array_equal(
            self.derivatives, np.ones_like(self.derivatives)
        ):
            return outputs.copy()
        inputs = np.empty_like(outputs)
        below = outputs < self.y_knots[0]
        above = outputs > self.y_knots[-1]
        inside = ~(below | above)
        inputs[below] = self.x_knots[0] + (
            outputs[below] - self.y_knots[0]
        ) / self.derivatives[0]
        inputs[above] = self.x_knots[-1] + (
            outputs[above] - self.y_knots[-1]
        ) / self.derivatives[-1]
        if np.any(inside):
            inputs[inside] = self._inverse_inside(outputs[inside])
        return inputs

    def derivative(self, values: np.ndarray) -> np.ndarray:
        inputs = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(inputs)):
            raise ValueError("spline inputs must be finite")
        if np.array_equal(self.x_knots, self.y_knots) and np.array_equal(
            self.derivatives, np.ones_like(self.derivatives)
        ):
            return np.ones_like(inputs)
        derivatives = np.empty_like(inputs)
        below = inputs < self.x_knots[0]
        above = inputs > self.x_knots[-1]
        inside = ~(below | above)
        derivatives[below] = self.derivatives[0]
        derivatives[above] = self.derivatives[-1]
        if np.any(inside):
            derivatives[inside] = self._forward_inside(inputs[inside])[1]
        return derivatives

    def _forward_inside(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bins = np.searchsorted(self.x_knots, values, side="right") - 1
        bins = np.clip(bins, 0, len(self.x_knots) - 2)
        x0 = self.x_knots[bins]
        y0 = self.y_knots[bins]
        widths = self.x_knots[bins + 1] - x0
        heights = self.y_knots[bins + 1] - y0
        slopes = heights / widths
        left_derivative = self.derivatives[bins]
        right_derivative = self.derivatives[bins + 1]
        theta = (values - x0) / widths
        theta_one_minus = theta * (1.0 - theta)
        denominator = slopes + (
            right_derivative + left_derivative - 2.0 * slopes
        ) * theta_one_minus
        numerator = heights * (
            slopes * theta**2 + left_derivative * theta_one_minus
        )
        outputs = y0 + numerator / denominator
        derivative_numerator = slopes**2 * (
            right_derivative * theta**2
            + 2.0 * slopes * theta_one_minus
            + left_derivative * (1.0 - theta) ** 2
        )
        return outputs, derivative_numerator / denominator**2

    def _inverse_inside(self, values: np.ndarray) -> np.ndarray:
        bins = np.searchsorted(self.y_knots, values, side="right") - 1
        bins = np.clip(bins, 0, len(self.y_knots) - 2)
        x0 = self.x_knots[bins]
        y0 = self.y_knots[bins]
        widths = self.x_knots[bins + 1] - x0
        heights = self.y_knots[bins + 1] - y0
        slopes = heights / widths
        left_derivative = self.derivatives[bins]
        right_derivative = self.derivatives[bins + 1]
        delta = values - y0
        curvature = right_derivative + left_derivative - 2.0 * slopes
        a = delta * curvature + heights * (slopes - left_derivative)
        b = heights * left_derivative - delta * curvature
        c = -slopes * delta
        discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)
        theta = np.where(
            np.abs(a) < 1e-14,
            -c / b,
            (2.0 * c) / (-b - np.sqrt(discriminant)),
        )
        return x0 + np.clip(theta, 0.0, 1.0) * widths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPLINE_SCHEMA,
            "x_knots": self.x_knots.tolist(),
            "y_knots": self.y_knots.tolist(),
            "derivatives": self.derivatives.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RationalQuadraticSpline":
        if payload.get("schema") != SPLINE_SCHEMA:
            raise ValueError(f"unsupported spline schema: {payload.get('schema')!r}")
        return cls(payload["x_knots"], payload["y_knots"], payload["derivatives"])


@dataclass(frozen=True)
class AffineMonotoneSplineOperator:
    """L2 operator: orientation-preserving affine followed by channel splines."""

    affine: AffineColorOperator
    splines: tuple[RationalQuadraticSpline, RationalQuadraticSpline, RationalQuadraticSpline]

    def __post_init__(self) -> None:
        if len(self.splines) != 3 or not all(
            isinstance(spline, RationalQuadraticSpline) for spline in self.splines
        ):
            raise ValueError("exactly three rational-quadratic splines are required")

    @classmethod
    def identity(cls, working_space: str = "linear_srgb") -> "AffineMonotoneSplineOperator":
        identity = RationalQuadraticSpline.identity()
        return cls(AffineColorOperator.identity(working_space), (identity, identity, identity))

    @property
    def working_space(self) -> str:
        return self.affine.working_space

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        intermediate = self.affine.apply(rgb)
        return np.stack(
            [self.splines[channel].apply(intermediate[..., channel]) for channel in range(3)],
            axis=-1,
        )

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        intermediate = np.stack(
            [self.splines[channel].inverse(values[..., channel]) for channel in range(3)],
            axis=-1,
        )
        return self.affine.inverse(intermediate)

    def jacobian_determinant(self, rgb: np.ndarray) -> np.ndarray:
        intermediate = self.affine.apply(rgb)
        spline_det = np.ones(intermediate.shape[:-1], dtype=np.float64)
        for channel, spline in enumerate(self.splines):
            spline_det *= spline.derivative(intermediate[..., channel])
        return self.affine.determinant * spline_det

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": L2_OPERATOR_SCHEMA,
            "working_space": self.working_space,
            "affine": self.affine.to_dict(),
            "splines": [spline.to_dict() for spline in self.splines],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AffineMonotoneSplineOperator":
        if payload.get("schema") != L2_OPERATOR_SCHEMA:
            raise ValueError(f"unsupported L2 operator schema: {payload.get('schema')!r}")
        affine = AffineColorOperator.from_dict(payload["affine"])
        if payload.get("working_space") != affine.working_space:
            raise ValueError("L2 operator working-space metadata is inconsistent")
        splines = tuple(RationalQuadraticSpline.from_dict(item) for item in payload["splines"])
        return cls(affine, splines)  # type: ignore[arg-type]
