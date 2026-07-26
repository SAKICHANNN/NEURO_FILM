"""Deterministic hard-strength preflight for an explicit colour operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class StrengthOperator(Protocol):
    def apply(self, linear_rgb: np.ndarray, *, strength: float) -> np.ndarray: ...


@dataclass(frozen=True)
class StrengthPreflightResult:
    selected_rgb8: np.ndarray
    challenger_rgb8: np.ndarray
    selected_strength: float
    challenger_new_hard_clipping_fraction: float
    fallback_applied: bool


def _rgb8(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 HxWx3 RGB")
    return array


def srgb8_to_linear(rgb8: np.ndarray) -> np.ndarray:
    encoded = _rgb8(rgb8, "rgb8").astype(np.float64) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb8(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or not np.all(np.isfinite(value))
        or np.any(value < -1e-12)
        or np.any(value > 1.0 + 1e-12)
    ):
        raise ValueError("linear RGB must be finite HxWx3 data in [0, 1]")
    bounded = np.clip(value, 0.0, 1.0)
    encoded = np.where(
        bounded <= 0.0031308,
        12.92 * bounded,
        1.055 * np.power(bounded, 1.0 / 2.4) - 0.055,
    )
    return np.rint(encoded * 255.0).astype(np.uint8)


def new_hard_clipping_fraction_rgb8(
    source_rgb8: np.ndarray,
    output_rgb8: np.ndarray,
    *,
    epsilon: float,
) -> float:
    source = _rgb8(source_rgb8, "source_rgb8")
    output = _rgb8(output_rgb8, "output_rgb8")
    if source.shape != output.shape:
        raise ValueError("source/output RGB8 dimensions differ")
    if not np.isfinite(epsilon) or epsilon < 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must be finite and in [0, 0.5)")
    source_values = source.astype(np.float64) / 255.0
    output_values = output.astype(np.float64) / 255.0
    source_endpoint = (source_values <= epsilon) | (
        source_values >= 1.0 - epsilon
    )
    output_endpoint = (output_values <= epsilon) | (
        output_values >= 1.0 - epsilon
    )
    return float(np.mean(output_endpoint & ~source_endpoint))


def render_strength_rgb8(
    source_rgb8: np.ndarray,
    operator: StrengthOperator,
    *,
    strength: float,
) -> np.ndarray:
    if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    linear = srgb8_to_linear(source_rgb8)
    rendered = operator.apply(linear, strength=float(strength))
    return linear_to_srgb8(rendered)


def apply_strength_preflight(
    source_rgb8: np.ndarray,
    operator: StrengthOperator,
    *,
    challenger_strength: float,
    baseline_strength: float,
    epsilon: float,
    maximum_new_hard_clipping_fraction: float,
) -> StrengthPreflightResult:
    """Trial the stronger explicit render and hard-fallback when necessary."""

    source = _rgb8(source_rgb8, "source_rgb8")
    if (
        not np.isfinite(maximum_new_hard_clipping_fraction)
        or maximum_new_hard_clipping_fraction < 0.0
        or maximum_new_hard_clipping_fraction > 1.0
    ):
        raise ValueError("maximum clipping fraction must be finite in [0, 1]")
    if baseline_strength >= challenger_strength:
        raise ValueError("baseline strength must be below challenger strength")
    challenger = render_strength_rgb8(
        source,
        operator,
        strength=challenger_strength,
    )
    clipping = new_hard_clipping_fraction_rgb8(
        source,
        challenger,
        epsilon=epsilon,
    )
    fallback = clipping > maximum_new_hard_clipping_fraction
    selected = (
        render_strength_rgb8(source, operator, strength=baseline_strength)
        if fallback
        else challenger
    )
    selected.setflags(write=False)
    challenger.setflags(write=False)
    return StrengthPreflightResult(
        selected_rgb8=selected,
        challenger_rgb8=challenger,
        selected_strength=(
            float(baseline_strength) if fallback else float(challenger_strength)
        ),
        challenger_new_hard_clipping_fraction=clipping,
        fallback_applied=fallback,
    )
