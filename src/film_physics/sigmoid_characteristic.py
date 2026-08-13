"""Explicit per-layer sigmoid characteristic curves fitted to a frozen prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.film_physics.bounded_dye_amount_direction import (
    apply_bounded_dye_amount_direction,
)
from src.film_physics.compact_log_scanner_compiler import (
    CompactLogScannerCompiler,
    apply_compact_log_scanner,
    invert_compact_log_scanner,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior


@dataclass(frozen=True)
class SigmoidCharacteristicCurve:
    lower: float
    upper: float
    slope: float
    midpoint: float

    def apply_normalized(self, coordinate: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinate, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("sigmoid characteristic coordinate must be in [0,1]")
        return self.lower + (self.upper - self.lower) / (
            1.0 + np.exp(-self.slope * (values - self.midpoint))
        )


def fit_sigmoid_characteristic(
    prior: ManufacturerCharacteristicPrior,
    *,
    samples: int,
    initial_slope: float,
    maximum_iterations: int,
) -> tuple[tuple[SigmoidCharacteristicCurve, ...], tuple[float, ...]]:
    if samples < 257 or initial_slope <= 0.0 or maximum_iterations < 100:
        raise ValueError("invalid sigmoid characteristic fit contract")
    coordinate = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    curves = []
    errors = []
    for source_curve in prior.curves:
        x0, x1 = source_curve.domain
        d0, d1 = source_curve.density_bounds
        target = source_curve.apply(x0 + coordinate * (x1 - x0))

        def residual(
            parameters: np.ndarray, target_values: np.ndarray = target
        ) -> np.ndarray:
            lower, log_span, log_slope, midpoint = parameters
            span = np.exp(log_span)
            slope = np.exp(log_slope)
            prediction = lower + span / (1.0 + np.exp(-slope * (coordinate - midpoint)))
            return prediction - target_values

        result = least_squares(
            residual,
            np.asarray((d0, np.log(d1 - d0), np.log(initial_slope), 0.5)),
            max_nfev=maximum_iterations,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError("sigmoid characteristic fit failed")
        lower, log_span, log_slope, midpoint = result.x
        curve = SigmoidCharacteristicCurve(
            float(lower),
            float(lower + np.exp(log_span)),
            float(np.exp(log_slope)),
            float(midpoint),
        )
        prediction = curve.apply_normalized(coordinate)
        curves.append(curve)
        errors.append(float(np.sqrt(np.mean((prediction - target) ** 2))))
    return tuple(curves), tuple(errors)


def render_sigmoid_scanner_positive(
    relative_display_linear: np.ndarray,
    structured_transmittance_rgb: np.ndarray,
    *,
    curves: tuple[SigmoidCharacteristicCurve, ...],
    compiler: CompactLogScannerCompiler,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.asarray(relative_display_linear, dtype=np.float64)
    structured = np.asarray(structured_transmittance_rgb)
    if (
        source.shape != structured.shape
        or source.ndim < 1
        or source.shape[-1] != 3
        or len(curves) != 3
        or structured.dtype != np.float32
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(structured))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(structured <= 0.0)
        or np.any(structured > 1.0)
    ):
        raise ValueError("invalid sigmoid scanner input")
    base_density = np.stack(
        [curve.apply_normalized(source[..., channel]) for channel, curve in enumerate(curves)],
        axis=-1,
    )
    candidate_density = -np.log10(structured.astype(np.float64))
    lower = np.asarray([curve.lower for curve in curves])
    upper = np.asarray([curve.upper for curve in curves])
    base_amount = (base_density - lower) / (upper - lower)
    candidate_amount = (candidate_density - lower) / (upper - lower)
    bounded, receipt = apply_bounded_dye_amount_direction(base_amount, candidate_amount)
    scan = apply_compact_log_scanner(
        np.ascontiguousarray(bounded[..., [2, 1, 0]], dtype=np.float32), compiler
    )
    positive = invert_compact_log_scanner(scan, compiler)[..., [2, 1, 0]]
    return np.ascontiguousarray(positive), receipt


__all__ = [
    "SigmoidCharacteristicCurve",
    "fit_sigmoid_characteristic",
    "render_sigmoid_scanner_positive",
]
