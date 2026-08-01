"""Compact positive-PSF representation for measured film MTF approximations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

BUNDLE_SCHEMA = "neuro_film.measured_positive_psf_bundle.v1"


@dataclass(frozen=True)
class PositivePsfComponent:
    weight: float
    sigma_um: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.weight)
            or not math.isfinite(self.sigma_um)
            or self.weight <= 0.0
            or self.sigma_um < 0.0
        ):
            raise ValueError("positive PSF component must be finite and nonnegative")


@dataclass(frozen=True)
class ChannelPsf:
    family: str
    components: tuple[PositivePsfComponent, ...]

    def __post_init__(self) -> None:
        if self.family not in {"single_gaussian", "delta_plus_gaussian", "two_gaussian"}:
            raise ValueError("unsupported positive PSF family")
        if not self.components or abs(sum(row.weight for row in self.components) - 1.0) > 1e-12:
            raise ValueError("positive PSF weights must sum to one")
        sigmas = tuple(row.sigma_um for row in self.components)
        if tuple(sorted(sigmas)) != sigmas:
            raise ValueError("positive PSF sigmas must be ordered")
        expected_count = 1 if self.family == "single_gaussian" else 2
        if len(self.components) != expected_count:
            raise ValueError("positive PSF component count does not match family")
        if self.family == "delta_plus_gaussian" and self.components[0].sigma_um != 0.0:
            raise ValueError("delta-plus-Gaussian requires an exact zero-sigma component")
        if self.family == "two_gaussian" and self.components[0].sigma_um <= 0.0:
            raise ValueError("two-Gaussian components must both have positive sigma")

    def response(self, frequencies_cycles_per_mm: Sequence[float]) -> np.ndarray:
        frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
            raise ValueError("MTF frequencies must be a finite nonnegative vector")
        result = np.zeros_like(frequencies)
        for component in self.components:
            sigma_mm = component.sigma_um * 1e-3
            result += component.weight * np.exp(
                -2.0 * math.pi**2 * sigma_mm**2 * frequencies**2
            )
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "components": [
                {"weight": row.weight, "sigma_um": row.sigma_um}
                for row in self.components
            ],
        }


def channel_psf_from_json(value: Mapping[str, Any]) -> ChannelPsf:
    components = tuple(
        PositivePsfComponent(weight=float(row["weight"]), sigma_um=float(row["sigma_um"]))
        for row in value["components"]
    )
    return ChannelPsf(family=str(value["family"]), components=components)


__all__ = [
    "BUNDLE_SCHEMA",
    "ChannelPsf",
    "PositivePsfComponent",
    "channel_psf_from_json",
]
