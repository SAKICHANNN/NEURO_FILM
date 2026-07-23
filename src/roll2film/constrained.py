"""Versioned numerical contract for global film-inspired colour operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .lut import DenseLUT3D, bake_dense_lut
from .operators import _validate_rgb
from .splines import AffineMonotoneSplineOperator


CONSTRAINT_SCHEMA = "roll2film.lut_constraint_spec.v1"
GLOBAL_OPERATOR_SCHEMA = "roll2film.constrained_global_operator.v1"


@dataclass(frozen=True)
class LUTConstraintSpec:
    output_minimum: float
    output_maximum: float
    maximum_residual_amplitude: float
    maximum_first_axis_step: float
    maximum_second_axis_difference: float
    maximum_neutral_axis_error: float
    minimum_tetrahedron_jacobian_determinant: float

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("LUT constraint values must be finite")
        if self.output_maximum <= self.output_minimum:
            raise ValueError("output maximum must exceed output minimum")
        if min(
            self.maximum_residual_amplitude,
            self.maximum_first_axis_step,
            self.maximum_second_axis_difference,
            self.maximum_neutral_axis_error,
        ) < 0.0:
            raise ValueError("LUT error and difference limits must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CONSTRAINT_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LUTConstraintSpec":
        values = dict(payload)
        if values.pop("schema", CONSTRAINT_SCHEMA) != CONSTRAINT_SCHEMA:
            raise ValueError("unsupported LUT constraint schema")
        return cls(**values)


@dataclass(frozen=True)
class LUTConstraintReport:
    output_minimum: float
    output_maximum: float
    maximum_residual_amplitude: float
    maximum_first_axis_step: float
    maximum_second_axis_difference: float
    maximum_neutral_axis_error: float
    minimum_tetrahedron_jacobian_determinant: float
    maximum_tetrahedron_jacobian_determinant: float
    output_range_pass: bool
    residual_pass: bool
    first_difference_pass: bool
    second_difference_pass: bool
    neutral_axis_pass: bool
    jacobian_pass: bool

    @property
    def passes(self) -> bool:
        return all(
            (
                self.output_range_pass,
                self.residual_pass,
                self.first_difference_pass,
                self.second_difference_pass,
                self.neutral_axis_pass,
                self.jacobian_pass,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passes": self.passes}


def identity_lut(
    size: int,
    *,
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> DenseLUT3D:
    return bake_dense_lut(
        _IdentityOperator(),
        size,
        domain_min=domain_min,
        domain_max=domain_max,
        interpolation="tetrahedral",
    )


def audit_lut_constraints(
    lut: DenseLUT3D,
    spec: LUTConstraintSpec,
) -> LUTConstraintReport:
    if lut.interpolation != "tetrahedral":
        raise ValueError("the constrained contract requires tetrahedral interpolation")
    axes = [
        np.linspace(lut.domain_min[channel], lut.domain_max[channel], lut.size)
        for channel in range(3)
    ]
    identity = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    residual = lut.values - identity
    first = [
        np.diff(residual, axis=axis)
        for axis in range(3)
    ]
    second = [
        np.diff(residual, n=2, axis=axis)
        for axis in range(3)
        if lut.size >= 3
    ]
    diagonal = lut.values[np.arange(lut.size), np.arange(lut.size), np.arange(lut.size)]
    neutral = np.stack(
        [np.linspace(lut.domain_min[channel], lut.domain_max[channel], lut.size) for channel in range(3)],
        axis=-1,
    )
    determinants = lut.tetrahedron_jacobian_determinants()

    output_minimum = float(np.min(lut.values))
    output_maximum = float(np.max(lut.values))
    maximum_residual = float(np.max(np.abs(residual)))
    maximum_first = float(max(np.max(np.abs(item)) for item in first))
    maximum_second = float(
        max(np.max(np.abs(item)) for item in second) if second else 0.0
    )
    maximum_neutral = float(np.max(np.abs(diagonal - neutral)))
    minimum_jacobian = float(np.min(determinants))
    maximum_jacobian = float(np.max(determinants))

    return LUTConstraintReport(
        output_minimum=output_minimum,
        output_maximum=output_maximum,
        maximum_residual_amplitude=maximum_residual,
        maximum_first_axis_step=maximum_first,
        maximum_second_axis_difference=maximum_second,
        maximum_neutral_axis_error=maximum_neutral,
        minimum_tetrahedron_jacobian_determinant=minimum_jacobian,
        maximum_tetrahedron_jacobian_determinant=maximum_jacobian,
        output_range_pass=(
            output_minimum >= spec.output_minimum
            and output_maximum <= spec.output_maximum
        ),
        residual_pass=maximum_residual <= spec.maximum_residual_amplitude,
        first_difference_pass=maximum_first <= spec.maximum_first_axis_step,
        second_difference_pass=maximum_second <= spec.maximum_second_axis_difference,
        neutral_axis_pass=maximum_neutral <= spec.maximum_neutral_axis_error,
        jacobian_pass=minimum_jacobian >= spec.minimum_tetrahedron_jacobian_determinant,
    )


@dataclass(frozen=True)
class ConstrainedGlobalColorOperator:
    base: AffineMonotoneSplineOperator
    lut: DenseLUT3D
    constraints: LUTConstraintSpec

    def __post_init__(self) -> None:
        report = audit_lut_constraints(self.lut, self.constraints)
        if not report.passes:
            raise ValueError(f"LUT violates the numerical contract: {report.to_dict()}")
        base_bounds = self.base.apply(_cube_corners())
        if np.any(base_bounds < self.lut.domain_min) or np.any(base_bounds > self.lut.domain_max):
            raise ValueError("base operator can leave the declared LUT domain")

    @classmethod
    def identity(
        cls,
        size: int,
        constraints: LUTConstraintSpec,
        working_space: str = "linear_srgb",
    ) -> "ConstrainedGlobalColorOperator":
        return cls(
            AffineMonotoneSplineOperator.identity(working_space),
            identity_lut(size),
            constraints,
        )

    @property
    def working_space(self) -> str:
        return self.base.working_space

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        values = _validate_rgb(rgb)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("global operator input falls outside [0, 1]")
        intermediate = self.base.apply(values)
        if np.any(intermediate < self.lut.domain_min) or np.any(intermediate > self.lut.domain_max):
            raise ValueError("base operator output falls outside the LUT domain")
        output = self.lut.apply(intermediate)
        if (
            np.any(output < self.constraints.output_minimum)
            or np.any(output > self.constraints.output_maximum)
        ):
            raise ValueError("global operator output falls outside declared headroom")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_OPERATOR_SCHEMA,
            "working_space": self.working_space,
            "base": self.base.to_dict(),
            "lut": self.lut.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConstrainedGlobalColorOperator":
        if payload.get("schema") != GLOBAL_OPERATOR_SCHEMA:
            raise ValueError("unsupported constrained global operator schema")
        base = AffineMonotoneSplineOperator.from_dict(payload["base"])
        if payload.get("working_space") != base.working_space:
            raise ValueError("global operator working-space metadata is inconsistent")
        return cls(
            base=base,
            lut=DenseLUT3D.from_dict(payload["lut"]),
            constraints=LUTConstraintSpec.from_dict(payload["constraints"]),
        )


class _IdentityOperator:
    @staticmethod
    def apply(rgb: np.ndarray) -> np.ndarray:
        return np.asarray(rgb, dtype=np.float64).copy()


def _cube_corners() -> np.ndarray:
    return np.asarray(
        [
            [red, green, blue]
            for red in (0.0, 1.0)
            for green in (0.0, 1.0)
            for blue in (0.0, 1.0)
        ],
        dtype=np.float64,
    )
