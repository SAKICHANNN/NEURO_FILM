"""Data-independent neutral-axis gauge for sensitometry-print operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import _validate_rgb
from .sensitometry_print import SensitometryPrintOperator
from .splines import RationalQuadraticSpline


NEUTRAL_GAUGE_SCHEMA = "roll2film.sensitometry_neutral_axis_gauge.v1"


@dataclass(frozen=True)
class NeutralAxisGaugeOperator:
    """Canonicalize a base operator through its own per-channel neutral response."""

    base: SensitometryPrintOperator
    inverse_neutral_splines: tuple[
        RationalQuadraticSpline,
        RationalQuadraticSpline,
        RationalQuadraticSpline,
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.base, SensitometryPrintOperator):
            raise ValueError("neutral-axis gauge requires SensitometryPrintOperator")
        if len(self.inverse_neutral_splines) != 3 or not all(
            isinstance(item, RationalQuadraticSpline)
            for item in self.inverse_neutral_splines
        ):
            raise ValueError("neutral-axis gauge requires three monotone splines")
        for spline in self.inverse_neutral_splines:
            if abs(float(spline.x_knots[0])) > 1e-12 or abs(float(spline.x_knots[-1]) - 1.0) > 1e-12:
                raise ValueError("neutral response must span exact normalized endpoints")
            if abs(float(spline.y_knots[0])) > 1e-12 or abs(float(spline.y_knots[-1]) - 1.0) > 1e-12:
                raise ValueError("inverse neutral gauge must span [0, 1]")

    @classmethod
    def from_base(cls, base: SensitometryPrintOperator, knot_count: int) -> "NeutralAxisGaugeOperator":
        if knot_count < 3:
            raise ValueError("neutral-axis gauge requires at least three knots")
        neutral = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
        response = base.apply(np.stack((neutral, neutral, neutral), axis=-1))
        splines = tuple(
            RationalQuadraticSpline.from_knots(response[:, channel], neutral)
            for channel in range(3)
        )
        return cls(base, splines)  # type: ignore[arg-type]

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(linear_rgb)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("neutral-axis gauge input must be finite [0, 1] RGB")
        intermediate = self.base.apply(values)
        output = np.stack(
            [
                self.inverse_neutral_splines[channel].apply(intermediate[..., channel])
                for channel in range(3)
            ],
            axis=-1,
        )
        if np.any(output < -1e-12) or np.any(output > 1.0 + 1e-12):
            raise RuntimeError("neutral-axis gauge escaped [0, 1]")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NEUTRAL_GAUGE_SCHEMA,
            "base": self.base.to_dict(),
            "inverse_neutral_splines": [item.to_dict() for item in self.inverse_neutral_splines],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeutralAxisGaugeOperator":
        if payload.get("schema") != NEUTRAL_GAUGE_SCHEMA:
            raise ValueError("unsupported neutral-axis gauge schema")
        splines = tuple(
            RationalQuadraticSpline.from_dict(item)
            for item in payload["inverse_neutral_splines"]
        )
        return cls(
            SensitometryPrintOperator.from_dict(payload["base"]),
            splines,  # type: ignore[arg-type]
        )
