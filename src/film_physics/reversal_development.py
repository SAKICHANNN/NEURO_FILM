"""Generic decreasing-density reversal development.

This candidate mirrors a frozen increasing-density curve family across its
exact endpoint densities. It is an explicit generic hypothesis, not measured
slide-film sensitometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from src.roll2film.sensitometry import RGBSensitometryOperator

from .contracts import PhysicalDomain, PhysicalDomainArray, PhysicalUnit
from .exposure_development import (
    DevelopedExposureResult,
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationEvidence,
    InterpretationRoute,
    ProcessCondition,
)


REVERSAL_DEVELOPMENT_SCHEMA = (
    "neuro_film.generic_density_mirrored_reversal.v1"
)
_CHANNELS = ("red", "green", "blue")


@dataclass(frozen=True)
class GenericReversalDevelopment:
    """Endpoint-mirrored, invertible decreasing-density operator."""

    template: RGBSensitometryOperator
    maximum_relative_layer_exposure: float

    def __post_init__(self) -> None:
        if not isinstance(self.template, RGBSensitometryOperator):
            raise TypeError("template must be RGBSensitometryOperator")
        maximum = float(self.maximum_relative_layer_exposure)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum layer exposure must be finite and positive")
        object.__setattr__(self, "maximum_relative_layer_exposure", maximum)

    @property
    def endpoint_density_sum(self) -> np.ndarray:
        endpoints = self.template.apply(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [self.maximum_relative_layer_exposure] * 3,
                ],
                dtype=np.float64,
            )
        )
        values = np.asarray(endpoints[0] + endpoints[1], dtype=np.float64)
        values.setflags(write=False)
        return values

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(linear_rgb, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 3 or values.size == 0:
            raise ValueError("layer exposure must have non-empty (..., 3) shape")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("layer exposure must be finite and nonnegative")
        if np.any(values > self.maximum_relative_layer_exposure):
            raise ValueError("layer exposure exceeds reversal maximum")
        density = self.endpoint_density_sum - self.template.apply(values)
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise RuntimeError("reversal development produced invalid density")
        return density

    def inverse(self, layer_density: np.ndarray) -> np.ndarray:
        values = np.asarray(layer_density, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 3 or values.size == 0:
            raise ValueError("layer density must have non-empty (..., 3) shape")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("layer density must be finite and nonnegative")
        template_density = self.endpoint_density_sum - values
        if np.any(template_density < 0.0):
            raise ValueError("layer density exceeds reversal domain")
        restored = self.template.inverse(template_density)
        if np.any(
            restored > self.maximum_relative_layer_exposure + np.float64(1e-12)
        ):
            raise ValueError("layer density is below reversal domain")
        return np.clip(restored, 0.0, self.maximum_relative_layer_exposure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REVERSAL_DEVELOPMENT_SCHEMA,
            "definition": "endpoint_density_sum_minus_template_density",
            "template": self.template.to_dict(),
            "maximum_relative_layer_exposure": (
                self.maximum_relative_layer_exposure
            ),
            "endpoint_density_sum": self.endpoint_density_sum.tolist(),
            "calibrated": False,
            "production_eligible": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenericReversalDevelopment":
        if payload.get("schema") != REVERSAL_DEVELOPMENT_SCHEMA:
            raise ValueError("unsupported reversal development schema")
        if (
            payload.get("definition")
            != "endpoint_density_sum_minus_template_density"
            or payload.get("calibrated") is not False
            or payload.get("production_eligible") is not False
        ):
            raise ValueError("unsupported reversal development declaration")
        candidate = cls(
            RGBSensitometryOperator.from_dict(payload["template"]),
            float(payload["maximum_relative_layer_exposure"]),
        )
        expected = candidate.endpoint_density_sum
        supplied = np.asarray(payload["endpoint_density_sum"], dtype=np.float64)
        if not np.array_equal(supplied, expected):
            raise ValueError("reversal endpoint density identity drift")
        return candidate


def reversal_development_identity(
    operator: GenericReversalDevelopment,
) -> str:
    if not isinstance(operator, GenericReversalDevelopment):
        raise TypeError("operator must be GenericReversalDevelopment")
    payload = json.dumps(
        operator.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reversal_development_contract(
    operator: GenericReversalDevelopment,
) -> DevelopmentInterpretationContract:
    """Bind the generic operator only to unknown-process direct slide."""
    return DevelopmentInterpretationContract(
        EmulsionFamily.SLIDE,
        InterpretationRoute.SLIDE_DIRECT_SCAN,
        ProcessCondition.UNKNOWN,
        InterpretationEvidence.UNKNOWN,
        reversal_development_identity(operator),
        operator.maximum_relative_layer_exposure,
    )


def develop_reversal_layer_exposure(
    exposure: PhysicalDomainArray,
    operator: GenericReversalDevelopment,
    contract: DevelopmentInterpretationContract,
) -> DevelopedExposureResult:
    """Apply the reversal operator across the typed development boundary."""
    if not isinstance(operator, GenericReversalDevelopment):
        raise TypeError("operator must be GenericReversalDevelopment")
    if not isinstance(contract, DevelopmentInterpretationContract):
        raise TypeError("contract must be DevelopmentInterpretationContract")
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != _CHANNELS:
        raise ValueError("v1 reversal development requires RGB layer order")
    if (
        contract.emulsion_family is not EmulsionFamily.SLIDE
        or contract.interpretation_route
        is not InterpretationRoute.SLIDE_DIRECT_SCAN
    ):
        raise ValueError("reversal development requires direct-slide contract")
    if contract.sensitometry_sha256 != reversal_development_identity(operator):
        raise ValueError("reversal development identity mismatch")
    if (
        contract.maximum_relative_layer_exposure
        != operator.maximum_relative_layer_exposure
    ):
        raise ValueError("reversal development maximum mismatch")
    values = operator.apply(exposure.values).astype(exposure.values.dtype)
    density = PhysicalDomainArray(
        values,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        exposure.channels,
        exposure.scale,
    )
    return DevelopedExposureResult(density, contract)
