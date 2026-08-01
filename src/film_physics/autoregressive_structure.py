"""Deterministic causal autoregressive Gaussian source fields.

This is a clean-room explicit-parameter research primitive.  It models only
spatial correlation; colour, density marginals and output interpretation stay
outside this module.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .structure_compiler import counter_normal_region


def causal_ar_positions(lag: int) -> tuple[tuple[int, int], ...]:
    """Return the standard causal 2D neighbourhood for ``lag`` 1..3."""
    if not isinstance(lag, int) or lag < 1 or lag > 3:
        raise ValueError("causal AR lag must be an integer in [1, 3]")
    positions = [
        (dy, dx)
        for dy in range(-lag, 0)
        for dx in range(-lag, lag + 1)
    ]
    positions.extend((0, dx) for dx in range(-lag, 0))
    output = tuple(positions)
    if len(output) != 2 * lag * (lag + 1):
        raise RuntimeError("causal AR neighbourhood size drift")
    return output


def quantize_ar_coefficients(
    coefficients: np.ndarray,
    *,
    step: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Quantize coefficients without clipping or silently repairing range."""
    values = np.asarray(coefficients, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not math.isfinite(step)
        or step <= 0.0
        or not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum >= maximum
        or np.any(values < minimum)
        or np.any(values > maximum)
    ):
        raise ValueError("AR coefficients are outside the frozen envelope")
    quantized = np.rint(values / step) * step
    if np.any(quantized < minimum) or np.any(quantized > maximum):
        raise ValueError("quantized AR coefficients left the frozen envelope")
    output = np.ascontiguousarray(quantized, dtype=np.float64)
    output.setflags(write=False)
    return output


def _validate_coefficients(coefficients: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != len(causal_ar_positions(lag))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("AR coefficient vector does not match its lag")
    return values


def filter_causal_ar(
    innovation: np.ndarray,
    *,
    lag: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Filter one finite 2D innovation field with zero boundary state."""
    source = np.asarray(innovation, dtype=np.float64)
    values = _validate_coefficients(coefficients, lag)
    if source.ndim != 2 or min(source.shape) <= lag or not np.all(np.isfinite(source)):
        raise ValueError("AR innovation must be finite 2D and larger than lag")
    positions = causal_ar_positions(lag)
    current = {
        -dx: float(values[index])
        for index, (dy, dx) in enumerate(positions)
        if dy == 0
    }
    denominator = np.ones(lag + 1, dtype=np.float64)
    for distance in range(1, lag + 1):
        denominator[distance] = -current[distance]
    output = np.empty_like(source)
    width = source.shape[1]
    for y in range(source.shape[0]):
        exogenous = source[y].copy()
        for coefficient, (dy, dx) in zip(values, positions, strict=True):
            if dy == 0 or y + dy < 0:
                continue
            previous = output[y + dy]
            if dx < 0:
                exogenous[-dx:] += coefficient * previous[: width + dx]
            elif dx > 0:
                exogenous[: width - dx] += coefficient * previous[dx:]
            else:
                exogenous += coefficient * previous
        output[y] = lfilter((1.0,), denominator, exogenous)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("causal AR field is non-finite")
    return output


def ar_impulse_metrics(
    coefficients: np.ndarray,
    *,
    lag: int,
    shape: tuple[int, int],
    tail_width: int,
) -> dict[str, float]:
    """Measure finite impulse energy and unresolved boundary-tail energy."""
    if (
        len(shape) != 2
        or min(shape) <= lag + 2
        or not isinstance(tail_width, int)
        or tail_width <= 0
        or tail_width * 2 >= min(shape)
    ):
        raise ValueError("invalid AR impulse metric geometry")
    impulse = np.zeros(shape, dtype=np.float64)
    impulse[0, 0] = 1.0
    response = filter_causal_ar(
        impulse,
        lag=lag,
        coefficients=coefficients,
    )
    energy = float(np.sum(np.square(response), dtype=np.float64))
    tail = np.zeros(shape, dtype=bool)
    tail[-tail_width:, :] = True
    tail[:, -tail_width:] = True
    tail_energy = float(np.sum(np.square(response[tail]), dtype=np.float64))
    if not math.isfinite(energy) or energy <= 0.0:
        raise RuntimeError("causal AR impulse energy is invalid")
    return {
        "energy": energy,
        "tail_energy_fraction": tail_energy / energy,
        "peak_absolute": float(np.max(np.abs(response))),
    }


def causal_ar_normal_region(
    full_shape: tuple[int, int],
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    lag: int,
    coefficients: np.ndarray,
    seed: int,
    normalization_energy: float,
) -> np.ndarray:
    """Render an exact finite region from a coordinate-seeded AR field.

    The research implementation deliberately recomputes the finite field from
    its zero-state origin.  This keeps arbitrary row partitions exact without
    inventing an unvalidated streaming-state contract.
    """
    height, width = full_shape
    y0, x0 = origin_yx
    region_height, region_width = shape
    if (
        height <= 0
        or width <= 0
        or y0 < 0
        or x0 < 0
        or region_height <= 0
        or region_width <= 0
        or y0 + region_height > height
        or x0 + region_width > width
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
        or not math.isfinite(normalization_energy)
        or normalization_energy <= 0.0
    ):
        raise ValueError("invalid causal AR region request")
    values = _validate_coefficients(coefficients, lag)
    innovation = counter_normal_region(
        full_shape,
        origin_yx=(0, 0),
        shape=full_shape,
        seed=seed,
    )
    field = filter_causal_ar(innovation, lag=lag, coefficients=values)
    field /= math.sqrt(normalization_energy)
    output = np.ascontiguousarray(
        field[y0 : y0 + region_height, x0 : x0 + region_width]
    )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("causal AR region is non-finite")
    output.setflags(write=False)
    return output


__all__ = [
    "ar_impulse_metrics",
    "causal_ar_normal_region",
    "causal_ar_positions",
    "filter_causal_ar",
    "quantize_ar_coefficients",
]
