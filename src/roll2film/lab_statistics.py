"""Paper-compatible, palette-sensitive Lab statistics for isolated research.

This module intentionally exposes statistics, not an operator estimator.  In
particular, spatial permutation invariance does not make the descriptor
content-independent or identify a colour transform outside observed support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.color import rgb2lab


LAB_STATISTICS_SCHEMA = "roll2film.paper_compatible_lab_statistics.v1"


def _readonly_float64(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array = np.array(array, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LabStatisticsDescriptor:
    """Immutable 2,304-D descriptor plus pre-square-root chroma audit mass."""

    lightness_histogram: np.ndarray
    chroma_histogram_sqrt: np.ndarray
    chroma_conditioned_mean_lightness: np.ndarray
    raw_chroma_histogram: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lightness_histogram",
            _readonly_float64(self.lightness_histogram, (256,), "lightness_histogram"),
        )
        object.__setattr__(
            self,
            "chroma_histogram_sqrt",
            _readonly_float64(self.chroma_histogram_sqrt, (32, 32), "chroma_histogram_sqrt"),
        )
        object.__setattr__(
            self,
            "chroma_conditioned_mean_lightness",
            _readonly_float64(
                self.chroma_conditioned_mean_lightness,
                (32, 32),
                "chroma_conditioned_mean_lightness",
            ),
        )
        object.__setattr__(
            self,
            "raw_chroma_histogram",
            _readonly_float64(self.raw_chroma_histogram, (32, 32), "raw_chroma_histogram"),
        )

    def vector(self) -> np.ndarray:
        result = np.concatenate(
            (
                self.lightness_histogram,
                self.chroma_histogram_sqrt.reshape(-1),
                self.chroma_conditioned_mean_lightness.reshape(-1),
            )
        )
        result.setflags(write=False)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LAB_STATISTICS_SCHEMA,
            "lightness_histogram": self.lightness_histogram.tolist(),
            "chroma_histogram_sqrt": self.chroma_histogram_sqrt.tolist(),
            "chroma_conditioned_mean_lightness": (
                self.chroma_conditioned_mean_lightness.tolist()
            ),
            "raw_chroma_histogram": self.raw_chroma_histogram.tolist(),
        }


def _soft_bin_1d(
    values: np.ndarray,
    *,
    bins: int,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(values, minimum, maximum)
    coordinate = (clipped - minimum) * ((bins - 1) / (maximum - minimum))
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, bins - 1)
    upper_weight = coordinate - lower
    lower_weight = 1.0 - upper_weight
    return lower, upper, lower_weight, upper_weight


def _accumulate_1d(
    lower: np.ndarray,
    upper: np.ndarray,
    lower_weight: np.ndarray,
    upper_weight: np.ndarray,
    bins: int,
) -> np.ndarray:
    result = np.zeros(bins, dtype=np.float64)
    np.add.at(result, lower, lower_weight)
    np.add.at(result, upper, upper_weight)
    return result


def _stretch_chroma(channel: np.ndarray, gamma: float, scale: float) -> np.ndarray:
    return np.sign(channel) * np.power(np.abs(channel) / scale, gamma)


def extract_lab_statistics(
    encoded_srgb: np.ndarray,
    *,
    chroma_gamma: float = 0.5,
    chroma_scale: float = 128.0,
    epsilon: float = 1e-12,
) -> LabStatisticsDescriptor:
    """Extract the StatLUT-style 2,304-D global Lab descriptor.

    Inputs are finite, encoded sRGB floats in ``[0, 1]`` with shape
    ``(..., 3)`` and at least one pixel.  No resize or spatial feature is used.
    """

    rgb = np.asarray(encoded_srgb, dtype=np.float64)
    if (
        rgb.ndim < 2
        or rgb.shape[-1] != 3
        or rgb.size == 0
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise ValueError("encoded_srgb must be finite [0, 1] data with shape (..., 3)")
    if chroma_gamma <= 0.0 or chroma_scale <= 0.0 or epsilon <= 0.0:
        raise ValueError("chroma parameters and epsilon must be positive")

    lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float64)
    lightness = lab[:, 0]
    stretched_a = np.clip(
        _stretch_chroma(lab[:, 1], chroma_gamma, chroma_scale),
        -1.0,
        1.0,
    )
    stretched_b = np.clip(
        _stretch_chroma(lab[:, 2], chroma_gamma, chroma_scale),
        -1.0,
        1.0,
    )
    pixel_count = lightness.size

    l0, l1, lw0, lw1 = _soft_bin_1d(
        lightness,
        bins=256,
        minimum=0.0,
        maximum=100.0,
    )
    lightness_histogram = _accumulate_1d(l0, l1, lw0, lw1, 256) / pixel_count

    a0, a1, aw0, aw1 = _soft_bin_1d(
        stretched_a,
        bins=32,
        minimum=-1.0,
        maximum=1.0,
    )
    b0, b1, bw0, bw1 = _soft_bin_1d(
        stretched_b,
        bins=32,
        minimum=-1.0,
        maximum=1.0,
    )
    raw_mass = np.zeros((32, 32), dtype=np.float64)
    lightness_mass = np.zeros((32, 32), dtype=np.float64)
    for ai, aw in ((a0, aw0), (a1, aw1)):
        for bi, bw in ((b0, bw0), (b1, bw1)):
            weight = aw * bw
            np.add.at(raw_mass, (ai, bi), weight)
            np.add.at(lightness_mass, (ai, bi), weight * lightness)
    raw_mass /= pixel_count
    lightness_mass /= pixel_count
    conditioned = np.divide(
        lightness_mass,
        raw_mass,
        out=np.zeros_like(lightness_mass),
        where=raw_mass > epsilon,
    )

    return LabStatisticsDescriptor(
        lightness_histogram=lightness_histogram,
        chroma_histogram_sqrt=np.sqrt(raw_mass),
        chroma_conditioned_mean_lightness=conditioned,
        raw_chroma_histogram=raw_mass,
    )
