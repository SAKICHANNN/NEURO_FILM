"""Bounded tetrahedral residual composition for explicit research operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constrained import LUTConstraintSpec, audit_lut_constraints
from .lut import DenseLUT3D
from .operators import _validate_rgb
from .sensitometry_print import SensitometryPrintOperator


SENSITOMETRY_RESIDUAL_SCHEMA = "roll2film.sensitometry_residual_lut.v1"


@dataclass(frozen=True)
class SensitometryResidualLUTOperator:
    """Apply one constrained display-domain LUT after sensitometry and print."""

    base: SensitometryPrintOperator
    lut: DenseLUT3D
    constraints: LUTConstraintSpec

    def __post_init__(self) -> None:
        if not isinstance(self.base, SensitometryPrintOperator):
            raise ValueError("residual composition requires SensitometryPrintOperator")
        if not isinstance(self.lut, DenseLUT3D):
            raise ValueError("residual composition requires DenseLUT3D")
        report = audit_lut_constraints(self.lut, self.constraints)
        if not report.passes:
            raise ValueError(f"residual LUT violates the numerical contract: {report.to_dict()}")
        if not np.array_equal(self.lut.domain_min, np.zeros(3)) or not np.array_equal(
            self.lut.domain_max, np.ones(3)
        ):
            raise ValueError("residual LUT domain must be exactly [0, 1]^3")

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(linear_rgb)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("residual composition input must be finite [0, 1] RGB")
        intermediate = self.base.apply(values)
        if np.any(intermediate < 0.0) or np.any(intermediate > 1.0):
            raise RuntimeError("sensitometry-print base escaped the residual LUT domain")
        output = self.lut.apply(intermediate)
        if np.any(output < self.constraints.output_minimum) or np.any(
            output > self.constraints.output_maximum
        ):
            raise RuntimeError("residual composition escaped declared output bounds")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SENSITOMETRY_RESIDUAL_SCHEMA,
            "base": self.base.to_dict(),
            "lut": self.lut.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensitometryResidualLUTOperator":
        if payload.get("schema") != SENSITOMETRY_RESIDUAL_SCHEMA:
            raise ValueError("unsupported sensitometry residual LUT schema")
        return cls(
            SensitometryPrintOperator.from_dict(payload["base"]),
            DenseLUT3D.from_dict(payload["lut"]),
            LUTConstraintSpec.from_dict(payload["constraints"]),
        )
