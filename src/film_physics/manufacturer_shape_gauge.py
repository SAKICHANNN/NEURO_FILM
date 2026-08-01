"""Compile manufacturer curve shapes into an explicit anchored hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.roll2film.sensitometry import (
    AnchoredCharacteristicCurve,
    RGBSensitometryOperator,
)
from src.roll2film.splines import RationalQuadraticSpline

from .manufacturer_characteristic import (
    CHANNELS,
    ManufacturerCharacteristicCurve,
    ManufacturerCharacteristicPrior,
)


@dataclass(frozen=True)
class ShapeGaugeChannel:
    layer: str
    source_anchor_log_exposure: float
    source_anchor_density: float
    generic_x_bounds: tuple[float, float]
    generic_y_bounds: tuple[float, float]
    strictness_repair_max_density: float
    mapped_x_knots: tuple[float, ...]
    mapped_y_knots: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "source_anchor_log_exposure": self.source_anchor_log_exposure,
            "source_anchor_density": self.source_anchor_density,
            "generic_x_bounds": list(self.generic_x_bounds),
            "generic_y_bounds": list(self.generic_y_bounds),
            "strictness_repair_max_density": self.strictness_repair_max_density,
            "mapped_x_knots": list(self.mapped_x_knots),
            "mapped_y_knots": list(self.mapped_y_knots),
        }


def _piecewise_map(
    value: np.ndarray,
    source: tuple[float, float, float],
    target: tuple[float, float, float],
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    source_min, source_anchor, source_max = source
    target_min, target_anchor, target_max = target
    if not source_min < source_anchor < source_max:
        raise ValueError("source gauge anchor must be strictly interior")
    left = target_min + (
        (values - source_min) / (source_anchor - source_min)
    ) * (target_anchor - target_min)
    right = target_anchor + (
        (values - source_anchor) / (source_max - source_anchor)
    ) * (target_max - target_anchor)
    return np.where(values <= source_anchor, left, right)


def _left_inverse(curve: ManufacturerCharacteristicCurve, density: float) -> float:
    x = curve.log_exposure_knots
    y = curve.density_knots
    if not float(y[0]) < density < float(y[-1]):
        raise ValueError("source density anchor must be strictly interior")
    index = int(np.searchsorted(y, density, side="left"))
    if y[index] == density:
        return float(x[index])
    left = index - 1
    fraction = (density - y[left]) / (y[index] - y[left])
    return float(x[left] + fraction * (x[index] - x[left]))


def _insert_anchor(
    x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float
) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.searchsorted(x, anchor_x, side="left"))
    if index < len(x) and x[index] == anchor_x:
        result_x = x.copy()
        result_y = y.copy()
        result_y[index] = anchor_y
        return result_x, result_y
    return np.insert(x, index, anchor_x), np.insert(y, index, anchor_y)


def compile_anchored_shape_gauge(
    manufacturer: ManufacturerCharacteristicPrior,
    generic: RGBSensitometryOperator,
    generic_config: Mapping[str, Any],
) -> tuple[RGBSensitometryOperator, tuple[ShapeGaugeChannel, ...]]:
    """Compile a no-fit gauge while preserving the generic encoder and ranges."""
    if tuple(curve.layer for curve in manufacturer.curves) != CHANNELS:
        raise ValueError("manufacturer channel order mismatch")
    generic_curves = {curve.layer: curve for curve in generic.curves}
    generic_x = np.asarray(generic_config["curve_x_knots"], dtype=np.float64)
    if not generic_x[0] < 0.0 < generic_x[-1]:
        raise ValueError("generic curve domain must contain the shared zero anchor")
    compiled = []
    metadata = []
    for source_curve in manufacturer.curves:
        layer = source_curve.layer
        generic_curve = generic_curves[layer]
        generic_y = np.asarray(generic_config["curve_y_knots"][layer], dtype=np.float64)
        generic_y_min = float(generic_y[0])
        generic_y_max = float(generic_y[-1])
        fraction = (generic_curve.anchor_density - generic_y_min) / (
            generic_y_max - generic_y_min
        )
        source_y_min, source_y_max = source_curve.density_bounds
        source_anchor_y = source_y_min + fraction * (source_y_max - source_y_min)
        source_anchor_x = _left_inverse(source_curve, source_anchor_y)
        source_x, source_y = _insert_anchor(
            source_curve.log_exposure_knots,
            source_curve.density_knots,
            source_anchor_x,
            source_anchor_y,
        )
        mapped_x = _piecewise_map(
            source_x,
            (float(source_x[0]), source_anchor_x, float(source_x[-1])),
            (float(generic_x[0]), 0.0, float(generic_x[-1])),
        )
        mapped_y_raw = _piecewise_map(
            source_y,
            (float(source_y[0]), source_anchor_y, float(source_y[-1])),
            (generic_y_min, generic_curve.anchor_density, generic_y_max),
        )
        anchor_index = int(np.argmin(np.abs(mapped_x)))
        mapped_x[anchor_index] = 0.0
        mapped_y_raw[anchor_index] = generic_curve.anchor_density
        mapped_y = mapped_y_raw.copy()
        for index in range(1, len(mapped_y)):
            if mapped_y[index] <= mapped_y[index - 1]:
                mapped_y[index] = np.nextafter(mapped_y[index - 1], np.inf)
        repair = float(np.max(np.abs(mapped_y - mapped_y_raw)))
        curve = AnchoredCharacteristicCurve(
            RationalQuadraticSpline.from_knots(mapped_x, mapped_y),
            0.0,
            generic_curve.anchor_density,
            layer,
        )
        compiled.append(curve)
        metadata.append(
            ShapeGaugeChannel(
                layer,
                source_anchor_x,
                source_anchor_y,
                (float(generic_x[0]), float(generic_x[-1])),
                (generic_y_min, generic_y_max),
                repair,
                tuple(map(float, mapped_x)),
                tuple(map(float, mapped_y)),
            )
        )
    return (
        RGBSensitometryOperator(
            generic.encoder,
            tuple(compiled),  # type: ignore[arg-type]
            "relative_layer_exposure_rgb_order",
            "status_m_layer_density",
        ),
        tuple(metadata),
    )


def mapped_source_target(
    source_curve: ManufacturerCharacteristicCurve,
    metadata: ShapeGaugeChannel,
    gauge_log_exposure: np.ndarray,
) -> np.ndarray:
    """Evaluate the piecewise-linear source target in gauge coordinates."""
    values = np.asarray(gauge_log_exposure, dtype=np.float64)
    source_x_min, source_x_max = source_curve.domain
    generic_x_min, generic_x_max = metadata.generic_x_bounds
    source_x = _piecewise_map(
        values,
        (generic_x_min, 0.0, generic_x_max),
        (source_x_min, metadata.source_anchor_log_exposure, source_x_max),
    )
    source_y = source_curve.apply(source_x)
    source_y_min, source_y_max = source_curve.density_bounds
    generic_y_min, generic_y_max = metadata.generic_y_bounds
    return _piecewise_map(
        source_y,
        (source_y_min, metadata.source_anchor_density, source_y_max),
        (generic_y_min, 1.0, generic_y_max),
    )


__all__ = [
    "ShapeGaugeChannel",
    "compile_anchored_shape_gauge",
    "mapped_source_target",
]
