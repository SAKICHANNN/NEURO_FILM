"""Compile developed density into the physical medium presented to a scanner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any

import numpy as np

from src.roll2film.sensitometry_print import DensityToPrintInterpretation

from .contracts import PhysicalDomain, PhysicalScale, density_to_transmittance
from .exposure_development import (
    DevelopedExposureResult,
    InterpretationRoute,
)


INTERPRETATION_MEDIUM_SCHEMA = "neuro_film.interpretation_medium.v1"
TRANSMITTANCE_OPERATOR_ID = "neuro-film-density-to-transmittance-v1"


class InterpretationMediumKind(str, Enum):
    FILM_TRANSMITTANCE = "film_transmittance"
    PRINT_REFLECTANCE = "print_reflectance"


class PostScanPolarity(str, Enum):
    IDENTITY = "identity"
    INVERT_TO_POSITIVE = "invert_to_positive"


def print_interpretation_identity(
    operator: DensityToPrintInterpretation,
) -> str:
    if not isinstance(operator, DensityToPrintInterpretation):
        raise TypeError("operator must be DensityToPrintInterpretation")
    encoded = json.dumps(
        operator.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class InterpretationMedium:
    values: np.ndarray
    kind: InterpretationMediumKind
    post_scan_polarity: PostScanPolarity
    interpretation_route: InterpretationRoute
    operator_identity: str
    channels: tuple[str, str, str]
    scale: PhysicalScale | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("interpretation medium must be float32 or float64")
        if values.ndim < 2 or values.shape[-1] != 3 or values.size == 0:
            raise ValueError("interpretation medium must have shape (..., 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("interpretation medium must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("interpretation medium must be bounded in [0, 1]")
        kind = InterpretationMediumKind(self.kind)
        polarity = PostScanPolarity(self.post_scan_polarity)
        route = InterpretationRoute(self.interpretation_route)
        channels = tuple(self.channels)
        if channels != ("red", "green", "blue"):
            raise ValueError("v1 interpretation medium requires RGB channel order")
        if not isinstance(self.operator_identity, str) or not self.operator_identity:
            raise ValueError("operator_identity must be non-empty")
        owned = np.array(values, copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "values", owned)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "post_scan_polarity", polarity)
        object.__setattr__(self, "interpretation_route", route)
        object.__setattr__(self, "channels", channels)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": INTERPRETATION_MEDIUM_SCHEMA,
            "kind": self.kind.value,
            "post_scan_polarity": self.post_scan_polarity.value,
            "interpretation_route": self.interpretation_route.value,
            "operator_identity": self.operator_identity,
            "channels": list(self.channels),
            "dtype": self.values.dtype.name,
            "shape": list(self.values.shape),
            "pixel_pitch_um": (
                None if self.scale is None else self.scale.pixel_pitch_um
            ),
        }


def prepare_interpretation_medium(
    developed: DevelopedExposureResult,
    *,
    print_interpretation: DensityToPrintInterpretation | None = None,
    bw_neutral_axis_tolerance: float = 1e-6,
) -> InterpretationMedium:
    """Create film transmittance or print reflectance without scanning it."""
    if not isinstance(developed, DevelopedExposureResult):
        raise TypeError("developed must be DevelopedExposureResult")
    density = developed.density.require(PhysicalDomain.DEVELOPED_DENSITY)
    route = developed.contract.interpretation_route
    if (
        not np.isfinite(bw_neutral_axis_tolerance)
        or bw_neutral_axis_tolerance < 0.0
    ):
        raise ValueError("B&W neutral-axis tolerance must be finite and nonnegative")

    if route is InterpretationRoute.COLOR_NEGATIVE_PRINT:
        if not isinstance(print_interpretation, DensityToPrintInterpretation):
            raise TypeError("print route requires DensityToPrintInterpretation")
        values = print_interpretation.apply(density.values)
        values = values.astype(density.values.dtype)
        kind = InterpretationMediumKind.PRINT_REFLECTANCE
        polarity = PostScanPolarity.IDENTITY
        identity = print_interpretation_identity(print_interpretation)
    else:
        if print_interpretation is not None:
            raise ValueError("film route cannot consume a print interpretation")
        if route is InterpretationRoute.BW_DEVELOPER_SCAN:
            neutral_error = float(
                np.max(np.ptp(density.values.astype(np.float64), axis=-1))
            )
            if neutral_error > bw_neutral_axis_tolerance:
                raise ValueError("B&W developed density is not on the neutral axis")
        transmittance = density_to_transmittance(density)
        values = transmittance.values
        kind = InterpretationMediumKind.FILM_TRANSMITTANCE
        polarity = (
            PostScanPolarity.IDENTITY
            if route is InterpretationRoute.SLIDE_DIRECT_SCAN
            else PostScanPolarity.INVERT_TO_POSITIVE
        )
        identity = TRANSMITTANCE_OPERATOR_ID
    return InterpretationMedium(
        values,
        kind,
        polarity,
        route,
        identity,
        density.channels,
        density.scale,
    )
