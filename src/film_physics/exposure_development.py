"""Typed reuse of U2.2 exposure-to-density sensitometry.

The module owns route and epistemic contracts only.  Characteristic-curve
arithmetic remains exclusively in :mod:`src.roll2film.sensitometry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any

import numpy as np

from src.roll2film.sensitometry import RGBSensitometryOperator

from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
)


DEVELOPMENT_CONTRACT_SCHEMA = (
    "neuro_film.physical_exposure_development_interpretation.v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHANNELS = ("red", "green", "blue")


class EmulsionFamily(str, Enum):
    COLOR_NEGATIVE = "color_negative"
    SLIDE = "slide"
    BLACK_AND_WHITE = "black_and_white"


class InterpretationRoute(str, Enum):
    COLOR_NEGATIVE_NEUTRAL_SCAN = "color_negative_neutral_scan"
    COLOR_NEGATIVE_PRINT = "color_negative_print"
    SLIDE_DIRECT_SCAN = "slide_direct_scan"
    BW_DEVELOPER_SCAN = "bw_developer_scan"


class ProcessCondition(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    PUSH = "push"
    PULL = "pull"


class InterpretationEvidence(str, Enum):
    UNKNOWN = "unknown"
    HYPOTHESIS_ONLY = "hypothesis_only"


_ALLOWED_ROUTES = {
    EmulsionFamily.COLOR_NEGATIVE: {
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        InterpretationRoute.COLOR_NEGATIVE_PRINT,
    },
    EmulsionFamily.SLIDE: {InterpretationRoute.SLIDE_DIRECT_SCAN},
    EmulsionFamily.BLACK_AND_WHITE: {InterpretationRoute.BW_DEVELOPER_SCAN},
}


def sensitometry_identity(operator: RGBSensitometryOperator) -> str:
    """Return a canonical identity without changing the U2.2 wire schema."""
    if not isinstance(operator, RGBSensitometryOperator):
        raise TypeError("operator must be RGBSensitometryOperator")
    encoded = json.dumps(
        operator.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DevelopmentInterpretationContract:
    emulsion_family: EmulsionFamily
    interpretation_route: InterpretationRoute
    process_condition: ProcessCondition
    interpretation_evidence: InterpretationEvidence
    sensitometry_sha256: str
    maximum_relative_layer_exposure: float = 16.0
    production_eligible: bool = False
    calibrated: bool = False

    def __post_init__(self) -> None:
        family = EmulsionFamily(self.emulsion_family)
        route = InterpretationRoute(self.interpretation_route)
        process = ProcessCondition(self.process_condition)
        evidence = InterpretationEvidence(self.interpretation_evidence)
        if route not in _ALLOWED_ROUTES[family]:
            raise ValueError("interpretation route does not belong to emulsion family")
        if process is ProcessCondition.UNKNOWN:
            if evidence is not InterpretationEvidence.UNKNOWN:
                raise ValueError("unknown process requires unknown evidence")
        elif evidence is not InterpretationEvidence.HYPOTHESIS_ONLY:
            raise ValueError("unmeasured process interpretation is hypothesis_only")
        if (
            not isinstance(self.sensitometry_sha256, str)
            or not _SHA256_RE.fullmatch(self.sensitometry_sha256)
        ):
            raise ValueError("sensitometry_sha256 must be lowercase hexadecimal")
        maximum = float(self.maximum_relative_layer_exposure)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum layer exposure must be finite and positive")
        if self.production_eligible is not False or self.calibrated is not False:
            raise ValueError("v1 generic development contract is research-only")
        object.__setattr__(self, "emulsion_family", family)
        object.__setattr__(self, "interpretation_route", route)
        object.__setattr__(self, "process_condition", process)
        object.__setattr__(self, "interpretation_evidence", evidence)
        object.__setattr__(self, "maximum_relative_layer_exposure", maximum)

    @classmethod
    def for_operator(
        cls,
        operator: RGBSensitometryOperator,
        *,
        emulsion_family: EmulsionFamily,
        interpretation_route: InterpretationRoute,
        process_condition: ProcessCondition = ProcessCondition.UNKNOWN,
        interpretation_evidence: InterpretationEvidence = (
            InterpretationEvidence.UNKNOWN
        ),
        maximum_relative_layer_exposure: float = 16.0,
    ) -> "DevelopmentInterpretationContract":
        return cls(
            emulsion_family,
            interpretation_route,
            process_condition,
            interpretation_evidence,
            sensitometry_identity(operator),
            maximum_relative_layer_exposure,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_CONTRACT_SCHEMA,
            "emulsion_family": self.emulsion_family.value,
            "interpretation_route": self.interpretation_route.value,
            "process_condition": self.process_condition.value,
            "interpretation_evidence": self.interpretation_evidence.value,
            "sensitometry_sha256": self.sensitometry_sha256,
            "maximum_relative_layer_exposure": (
                self.maximum_relative_layer_exposure
            ),
            "production_eligible": self.production_eligible,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "DevelopmentInterpretationContract":
        if payload.get("schema") != DEVELOPMENT_CONTRACT_SCHEMA:
            raise ValueError("unsupported development contract schema")
        return cls(
            EmulsionFamily(payload["emulsion_family"]),
            InterpretationRoute(payload["interpretation_route"]),
            ProcessCondition(payload["process_condition"]),
            InterpretationEvidence(payload["interpretation_evidence"]),
            str(payload["sensitometry_sha256"]),
            float(payload["maximum_relative_layer_exposure"]),
            payload["production_eligible"],
            payload["calibrated"],
        )


@dataclass(frozen=True)
class DevelopedExposureResult:
    density: PhysicalDomainArray
    contract: DevelopmentInterpretationContract

    def __post_init__(self) -> None:
        self.density.require(PhysicalDomain.DEVELOPED_DENSITY)
        if not isinstance(self.contract, DevelopmentInterpretationContract):
            raise TypeError("contract must be DevelopmentInterpretationContract")


def develop_layer_exposure(
    exposure: PhysicalDomainArray,
    operator: RGBSensitometryOperator,
    contract: DevelopmentInterpretationContract,
) -> DevelopedExposureResult:
    """Apply the exact U2.2 operator across a typed physical boundary."""
    if not isinstance(contract, DevelopmentInterpretationContract):
        raise TypeError("contract must be DevelopmentInterpretationContract")
    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.channels != _CHANNELS:
        raise ValueError("v1 development requires red/green/blue layer order")
    if sensitometry_identity(operator) != contract.sensitometry_sha256:
        raise ValueError("sensitometry identity does not match contract")
    if float(np.max(exposure.values)) > contract.maximum_relative_layer_exposure:
        raise ValueError("layer exposure exceeds the frozen contract maximum")
    values = operator.apply(exposure.values)
    dtype = exposure.values.dtype
    density = PhysicalDomainArray(
        values.astype(dtype),
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        exposure.channels,
        exposure.scale,
    )
    return DevelopedExposureResult(density, contract)
