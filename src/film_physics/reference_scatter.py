"""Float64 full-frame reference for exposure-domain film scatter.

The implementation is intentionally global and expensive. It defines a
numerical truth for later separable/pyramid runtime challengers; it is not a
product renderer or a calibrated stock profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from .contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)


REFERENCE_SCATTER_SCHEMA = "neuro_film.reference_exposure_scatter.v1"


@dataclass(frozen=True)
class ScatterComponent:
    component_id: str
    sigma_um: float
    cutoff_sigma: float
    energy_fraction_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("scatter component_id must be non-empty")
        sigma = float(self.sigma_um)
        cutoff = float(self.cutoff_sigma)
        fractions = tuple(float(value) for value in self.energy_fraction_rgb)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("scatter sigma_um must be finite and positive")
        if not math.isfinite(cutoff) or cutoff < 3.0 or cutoff > 8.0:
            raise ValueError("scatter cutoff_sigma must be in [3, 8]")
        if len(fractions) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in fractions
        ):
            raise ValueError("scatter energy fractions must be three finite [0, 1] values")
        object.__setattr__(self, "sigma_um", sigma)
        object.__setattr__(self, "cutoff_sigma", cutoff)
        object.__setattr__(self, "energy_fraction_rgb", fractions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "sigma_um": self.sigma_um,
            "cutoff_sigma": self.cutoff_sigma,
            "energy_fraction_rgb": list(self.energy_fraction_rgb),
        }


@dataclass(frozen=True)
class ReferenceScatterProfile:
    pixel_pitch_um: float
    components: tuple[ScatterComponent, ...]
    boundary_mode: str = "zero-outside-full-frame"
    schema: str = REFERENCE_SCATTER_SCHEMA

    def __post_init__(self) -> None:
        scale = PhysicalScale(self.pixel_pitch_um)
        components = tuple(self.components)
        if self.schema != REFERENCE_SCATTER_SCHEMA:
            raise ValueError("unsupported reference scatter schema")
        if self.boundary_mode != "zero-outside-full-frame":
            raise ValueError("v1 reference scatter requires zero-outside-full-frame")
        if not components or not all(
            isinstance(component, ScatterComponent) for component in components
        ):
            raise ValueError("reference scatter requires typed components")
        if len({component.component_id for component in components}) != len(components):
            raise ValueError("scatter component IDs must be unique")
        totals = np.sum(
            np.asarray(
                [component.energy_fraction_rgb for component in components],
                dtype=np.float64,
            ),
            axis=0,
        )
        if np.any(totals > 1.0):
            raise ValueError("scatter energy fractions exceed incident energy")
        object.__setattr__(self, "pixel_pitch_um", scale.pixel_pitch_um)
        object.__setattr__(self, "components", components)

    @property
    def scale(self) -> PhysicalScale:
        return PhysicalScale(self.pixel_pitch_um)

    @property
    def total_scatter_fraction(self) -> np.ndarray:
        return np.sum(
            np.asarray(
                [component.energy_fraction_rgb for component in self.components],
                dtype=np.float64,
            ),
            axis=0,
        )

    @property
    def direct_fraction(self) -> np.ndarray:
        return 1.0 - self.total_scatter_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pixel_pitch_um": self.pixel_pitch_um,
            "boundary_mode": self.boundary_mode,
            "components": [component.to_dict() for component in self.components],
        }

    @property
    def profile_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def profile_from_contract(contract: dict[str, Any]) -> ReferenceScatterProfile:
    if contract.get("schema") != (
        "neuro_film.u6_p1_reference_scatter_simulator_contract.v1"
    ):
        raise ValueError("unsupported U6.P1 scatter contract")
    if contract.get("input_domain") != PhysicalDomain.LAYER_EXPOSURE.value:
        raise ValueError("U6.P1 input domain mismatch")
    if contract.get("output_domain") != PhysicalDomain.LAYER_EXPOSURE.value:
        raise ValueError("U6.P1 output domain mismatch")
    if contract.get("arithmetic") != "float64":
        raise ValueError("U6.P1 reference arithmetic must be float64")
    components = tuple(
        ScatterComponent(
            component_id=str(item["component_id"]),
            sigma_um=float(item["sigma_um"]),
            cutoff_sigma=float(item["cutoff_sigma"]),
            energy_fraction_rgb=tuple(item["energy_fraction_rgb"]),  # type: ignore[arg-type]
        )
        for item in contract["components"]
    )
    return ReferenceScatterProfile(
        pixel_pitch_um=float(contract["pixel_pitch_um"]),
        boundary_mode=str(contract["boundary_mode"]),
        components=components,
    )


def gaussian_kernel_2d(
    component: ScatterComponent, scale: PhysicalScale
) -> np.ndarray:
    sigma_px = component.sigma_um / scale.pixel_pitch_um
    radius = max(1, int(math.ceil(component.cutoff_sigma * sigma_px)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * np.square(coordinates / sigma_px))
    kernel_1d /= np.sum(kernel_1d, dtype=np.float64)
    kernel = np.multiply.outer(kernel_1d, kernel_1d)
    kernel /= np.sum(kernel, dtype=np.float64)
    kernel.setflags(write=False)
    return kernel


def apply_reference_scatter(
    exposure: PhysicalDomainArray,
    profile: ReferenceScatterProfile,
) -> PhysicalDomainArray:
    """Apply energy-partitioned full-frame scatter without clipping."""

    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float64:
        raise TypeError("reference scatter requires float64 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("reference scatter requires an HxWx3 image")
    if exposure.scale != profile.scale:
        raise ValueError("reference scatter pixel scale does not match the profile")
    values = exposure.values
    output = values * profile.direct_fraction.reshape(1, 1, 3)
    for component in profile.components:
        kernel = gaussian_kernel_2d(component, profile.scale)
        weights = np.asarray(component.energy_fraction_rgb, dtype=np.float64)
        for channel in range(3):
            if weights[channel] == 0.0:
                continue
            blurred = fftconvolve(values[..., channel], kernel, mode="same")
            output[..., channel] += weights[channel] * blurred
    roundoff_floor = -64.0 * np.finfo(np.float64).eps
    if float(np.min(output)) < roundoff_floor:
        raise RuntimeError("reference scatter produced materially negative exposure")
    output = np.maximum(output, 0.0)
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )
