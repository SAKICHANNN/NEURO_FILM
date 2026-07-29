"""Float64 reference for bounded backing-reflection layer re-exposure.

Unlike :mod:`reference_scatter`, this operator does not subtract returned
energy from the direct exposure. Light that reaches a support surface can
return through the emulsion and create a second exposure. The returned term is
therefore additive, while its energy remains bounded by explicit nonnegative
per-source fractions and coupling columns.

This is a generic physical-inspired numerical reference. Its profile values
are not calibrated to a named film, camera, pressure plate, or process.
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


BACKING_RETURN_SCHEMA = "neuro_film.reference_backing_return.v1"


@dataclass(frozen=True)
class BackingReturnComponent:
    component_id: str
    sigma_um: float
    cutoff_sigma: float
    return_fraction_rgb: tuple[float, float, float]
    source_to_layer_coupling: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("backing-return component_id must be non-empty")
        sigma = float(self.sigma_um)
        cutoff = float(self.cutoff_sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("backing-return sigma_um must be finite and positive")
        if not math.isfinite(cutoff) or cutoff < 3.0 or cutoff > 8.0:
            raise ValueError("backing-return cutoff_sigma must be in [3, 8]")

        fractions = tuple(float(value) for value in self.return_fraction_rgb)
        if len(fractions) != 3 or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in fractions
        ):
            raise ValueError(
                "return fractions must contain three finite values in [0, 1]"
            )

        coupling = np.asarray(self.source_to_layer_coupling, dtype=np.float64)
        if coupling.shape != (3, 3):
            raise ValueError("source-to-layer coupling must be a 3x3 matrix")
        if not np.all(np.isfinite(coupling)) or np.any(coupling < 0.0):
            raise ValueError("source-to-layer coupling must be finite and nonnegative")
        if np.any(np.sum(coupling, axis=0, dtype=np.float64) > 1.0 + 1e-15):
            raise ValueError("source-to-layer coupling column exceeds unity")

        canonical_coupling = tuple(
            tuple(float(value) for value in row) for row in coupling
        )
        object.__setattr__(self, "sigma_um", sigma)
        object.__setattr__(self, "cutoff_sigma", cutoff)
        object.__setattr__(self, "return_fraction_rgb", fractions)
        object.__setattr__(self, "source_to_layer_coupling", canonical_coupling)

    @property
    def coupling_array(self) -> np.ndarray:
        return np.asarray(self.source_to_layer_coupling, dtype=np.float64)

    @property
    def return_weights(self) -> np.ndarray:
        """Target-layer by source-channel returned-energy weights."""

        return self.coupling_array * np.asarray(
            self.return_fraction_rgb, dtype=np.float64
        ).reshape(1, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "sigma_um": self.sigma_um,
            "cutoff_sigma": self.cutoff_sigma,
            "return_fraction_rgb": list(self.return_fraction_rgb),
            "source_to_layer_coupling": [
                list(row) for row in self.source_to_layer_coupling
            ],
        }


@dataclass(frozen=True)
class BackingReturnProfile:
    pixel_pitch_um: float
    components: tuple[BackingReturnComponent, ...]
    boundary_mode: str = "zero-outside-full-frame"
    schema: str = BACKING_RETURN_SCHEMA

    def __post_init__(self) -> None:
        scale = PhysicalScale(self.pixel_pitch_um)
        components = tuple(self.components)
        if self.schema != BACKING_RETURN_SCHEMA:
            raise ValueError("unsupported backing-return schema")
        if self.boundary_mode != "zero-outside-full-frame":
            raise ValueError("v1 backing return requires zero-outside-full-frame")
        if not components or not all(
            isinstance(component, BackingReturnComponent)
            for component in components
        ):
            raise ValueError("backing return requires typed components")
        if len({component.component_id for component in components}) != len(
            components
        ):
            raise ValueError("backing-return component IDs must be unique")

        fractions = np.sum(
            np.asarray(
                [component.return_fraction_rgb for component in components],
                dtype=np.float64,
            ),
            axis=0,
        )
        if np.any(fractions > 1.0 + 1e-15):
            raise ValueError("aggregate backing-return fraction exceeds incident energy")
        object.__setattr__(self, "pixel_pitch_um", scale.pixel_pitch_um)
        object.__setattr__(self, "components", components)

    @property
    def scale(self) -> PhysicalScale:
        return PhysicalScale(self.pixel_pitch_um)

    @property
    def maximum_return_fraction_by_source(self) -> np.ndarray:
        """Maximum total returned layer energy for each source channel."""

        total = np.zeros(3, dtype=np.float64)
        for component in self.components:
            total += np.sum(component.return_weights, axis=0, dtype=np.float64)
        return total

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


def backing_return_profile_from_contract(
    contract: dict[str, Any],
) -> BackingReturnProfile:
    if contract.get("schema") != (
        "neuro_film.u6_p3d_backing_return_reference_contract.v1"
    ):
        raise ValueError("unsupported U6.P3D backing-return contract")
    if contract.get("input_domain") != PhysicalDomain.LAYER_EXPOSURE.value:
        raise ValueError("U6.P3D input domain mismatch")
    if contract.get("output_domain") != PhysicalDomain.LAYER_EXPOSURE.value:
        raise ValueError("U6.P3D output domain mismatch")
    if contract.get("arithmetic") != "float64":
        raise ValueError("U6.P3D reference arithmetic must be float64")
    components = tuple(
        BackingReturnComponent(
            component_id=str(item["component_id"]),
            sigma_um=float(item["sigma_um"]),
            cutoff_sigma=float(item["cutoff_sigma"]),
            return_fraction_rgb=tuple(item["return_fraction_rgb"]),  # type: ignore[arg-type]
            source_to_layer_coupling=tuple(  # type: ignore[arg-type]
                tuple(row) for row in item["source_to_layer_coupling"]
            ),
        )
        for item in contract["components"]
    )
    return BackingReturnProfile(
        pixel_pitch_um=float(contract["pixel_pitch_um"]),
        boundary_mode=str(contract["boundary_mode"]),
        components=components,
    )


def backing_return_kernel_1d(
    component: BackingReturnComponent, scale: PhysicalScale
) -> np.ndarray:
    sigma_px = component.sigma_um / scale.pixel_pitch_um
    radius = max(1, int(math.ceil(component.cutoff_sigma * sigma_px)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / sigma_px))
    kernel /= np.sum(kernel, dtype=np.float64)
    kernel.setflags(write=False)
    return kernel


def backing_return_kernel_2d(
    component: BackingReturnComponent, scale: PhysicalScale
) -> np.ndarray:
    one_dimensional = backing_return_kernel_1d(component, scale)
    kernel = np.multiply.outer(one_dimensional, one_dimensional)
    kernel /= np.sum(kernel, dtype=np.float64)
    kernel.setflags(write=False)
    return kernel


def apply_reference_backing_return(
    exposure: PhysicalDomainArray,
    profile: BackingReturnProfile,
) -> PhysicalDomainArray:
    """Add bounded, nonnegative support-reflected second-pass exposure."""

    exposure.require(PhysicalDomain.LAYER_EXPOSURE)
    if exposure.values.dtype != np.float64:
        raise TypeError("reference backing return requires float64 exposure")
    if exposure.values.ndim != 3:
        raise ValueError("reference backing return requires an HxWx3 image")
    if exposure.scale != profile.scale:
        raise ValueError("backing-return pixel scale does not match the profile")

    values = exposure.values
    output = np.array(values, dtype=np.float64, copy=True, order="C")
    for component in profile.components:
        kernel = backing_return_kernel_2d(component, profile.scale)
        weights = component.return_weights
        for source_channel in range(3):
            if not np.any(weights[:, source_channel]):
                continue
            returned = fftconvolve(
                values[..., source_channel], kernel, mode="same"
            )
            for target_layer in range(3):
                weight = weights[target_layer, source_channel]
                if weight != 0.0:
                    output[..., target_layer] += weight * returned

    roundoff_floor = -64.0 * np.finfo(np.float64).eps
    if float(np.min(output)) < roundoff_floor:
        raise RuntimeError("reference backing return produced negative exposure")
    output = np.maximum(output, 0.0)
    return PhysicalDomainArray(
        output,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        exposure.channels,
        exposure.scale,
    )
