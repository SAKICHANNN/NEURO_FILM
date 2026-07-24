"""Explicit monotone linear-exposure to layer-density sensitometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import _validate_rgb
from .splines import RationalQuadraticSpline


EXPOSURE_ENCODER_SCHEMA = "roll2film.log_exposure_encoder.v1"
CHARACTERISTIC_CURVE_SCHEMA = "roll2film.anchored_characteristic_curve.v1"
SENSITOMETRY_SCHEMA = "roll2film.rgb_sensitometry.v1"
_ANCHOR_TOLERANCE = 1e-12
_BOUNDARY_ROUNDOFF_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LogExposureEncoder:
    """Invertible nonnegative linear exposure to relative log10 exposure."""

    reference_linear: float
    black_offset: float

    def __post_init__(self) -> None:
        reference = float(self.reference_linear)
        offset = float(self.black_offset)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference linear exposure must be finite and positive")
        if not np.isfinite(offset) or offset <= 0.0:
            raise ValueError("black offset must be finite and positive")
        object.__setattr__(self, "reference_linear", reference)
        object.__setattr__(self, "black_offset", offset)

    @property
    def minimum_log_exposure(self) -> float:
        return float(
            np.log10(self.black_offset / (self.reference_linear + self.black_offset))
        )

    def apply(self, linear: np.ndarray) -> np.ndarray:
        values = np.asarray(linear, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("linear exposure must be finite")
        if np.any(values < 0.0):
            raise ValueError("linear exposure must be nonnegative")
        return np.log10(
            (values + self.black_offset) / (self.reference_linear + self.black_offset)
        )

    def inverse(self, log_exposure: np.ndarray) -> np.ndarray:
        values = np.asarray(log_exposure, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("log exposure must be finite")
        boundary = self.minimum_log_exposure
        if np.any(values < boundary - _BOUNDARY_ROUNDOFF_TOLERANCE):
            raise ValueError("log exposure is below the zero-exposure boundary")
        values = np.where(values < boundary, boundary, values)
        linear = (self.reference_linear + self.black_offset) * np.power(10.0, values)
        linear -= self.black_offset
        if np.any(linear < -1e-15):
            raise ValueError("inverse exposure produced a negative value")
        return linear

    def derivative(self, linear: np.ndarray) -> np.ndarray:
        values = np.asarray(linear, dtype=np.float64)
        self.apply(values)
        return 1.0 / (np.log(10.0) * (values + self.black_offset))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPOSURE_ENCODER_SCHEMA,
            "reference_linear": self.reference_linear,
            "black_offset": self.black_offset,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogExposureEncoder":
        if payload.get("schema") != EXPOSURE_ENCODER_SCHEMA:
            raise ValueError(f"unsupported exposure encoder schema: {payload.get('schema')!r}")
        return cls(float(payload["reference_linear"]), float(payload["black_offset"]))


@dataclass(frozen=True)
class AnchoredCharacteristicCurve:
    """Strictly increasing log-exposure to density characteristic curve."""

    spline: RationalQuadraticSpline
    anchor_log_exposure: float
    anchor_density: float
    layer: str

    def __post_init__(self) -> None:
        if not isinstance(self.spline, RationalQuadraticSpline):
            raise ValueError("characteristic curve requires a rational-quadratic spline")
        anchor_x = float(self.anchor_log_exposure)
        anchor_y = float(self.anchor_density)
        if not np.isfinite(anchor_x) or not np.isfinite(anchor_y):
            raise ValueError("characteristic anchor must be finite")
        if not isinstance(self.layer, str) or not self.layer:
            raise ValueError("characteristic layer must be a non-empty string")
        actual = float(self.spline.apply(np.asarray(anchor_x)))
        if abs(actual - anchor_y) > _ANCHOR_TOLERANCE:
            raise ValueError("characteristic curve does not satisfy its neutral anchor")
        object.__setattr__(self, "anchor_log_exposure", anchor_x)
        object.__setattr__(self, "anchor_density", anchor_y)

    def apply(self, log_exposure: np.ndarray) -> np.ndarray:
        return self.spline.apply(log_exposure)

    def inverse(self, density: np.ndarray) -> np.ndarray:
        return self.spline.inverse(density)

    def derivative(self, log_exposure: np.ndarray) -> np.ndarray:
        return self.spline.derivative(log_exposure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHARACTERISTIC_CURVE_SCHEMA,
            "layer": self.layer,
            "anchor_log_exposure": self.anchor_log_exposure,
            "anchor_density": self.anchor_density,
            "spline": self.spline.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchoredCharacteristicCurve":
        if payload.get("schema") != CHARACTERISTIC_CURVE_SCHEMA:
            raise ValueError(
                f"unsupported characteristic curve schema: {payload.get('schema')!r}"
            )
        return cls(
            RationalQuadraticSpline.from_dict(payload["spline"]),
            float(payload["anchor_log_exposure"]),
            float(payload["anchor_density"]),
            str(payload["layer"]),
        )


@dataclass(frozen=True)
class RGBSensitometryOperator:
    """Three-channel exposure encoder and layer characteristic curves."""

    encoder: LogExposureEncoder
    curves: tuple[
        AnchoredCharacteristicCurve,
        AnchoredCharacteristicCurve,
        AnchoredCharacteristicCurve,
    ]
    input_space: str = "linear_srgb"
    output_space: str = "layer_density"

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, LogExposureEncoder):
            raise ValueError("sensitometry operator requires a LogExposureEncoder")
        if len(self.curves) != 3 or not all(
            isinstance(curve, AnchoredCharacteristicCurve) for curve in self.curves
        ):
            raise ValueError("sensitometry operator requires exactly three curves")
        if tuple(curve.layer for curve in self.curves) != ("red", "green", "blue"):
            raise ValueError("sensitometry curves must be ordered red, green, blue")
        anchors = {(curve.anchor_log_exposure, curve.anchor_density) for curve in self.curves}
        if len(anchors) != 1:
            raise ValueError("sensitometry curves must share one neutral anchor")
        if not self.input_space or not self.output_space:
            raise ValueError("sensitometry spaces must be non-empty")

    def log_exposure(self, linear_rgb: np.ndarray) -> np.ndarray:
        return self.encoder.apply(_validate_rgb(linear_rgb))

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        exposure = self.log_exposure(linear_rgb)
        return np.stack(
            [self.curves[index].apply(exposure[..., index]) for index in range(3)],
            axis=-1,
        )

    def inverse(self, layer_density: np.ndarray) -> np.ndarray:
        density = _validate_rgb(layer_density)
        exposure = np.stack(
            [self.curves[index].inverse(density[..., index]) for index in range(3)],
            axis=-1,
        )
        return self.encoder.inverse(exposure)

    def jacobian_determinant(self, linear_rgb: np.ndarray) -> np.ndarray:
        linear = _validate_rgb(linear_rgb)
        exposure = self.encoder.apply(linear)
        determinant = np.ones(linear.shape[:-1], dtype=np.float64)
        encoder_derivative = self.encoder.derivative(linear)
        for index, curve in enumerate(self.curves):
            determinant *= curve.derivative(exposure[..., index])
            determinant *= encoder_derivative[..., index]
        return determinant

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SENSITOMETRY_SCHEMA,
            "input_space": self.input_space,
            "output_space": self.output_space,
            "encoder": self.encoder.to_dict(),
            "curves": [curve.to_dict() for curve in self.curves],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RGBSensitometryOperator":
        if payload.get("schema") != SENSITOMETRY_SCHEMA:
            raise ValueError(f"unsupported sensitometry schema: {payload.get('schema')!r}")
        curves = tuple(
            AnchoredCharacteristicCurve.from_dict(item) for item in payload["curves"]
        )
        return cls(
            LogExposureEncoder.from_dict(payload["encoder"]),
            curves,  # type: ignore[arg-type]
            str(payload["input_space"]),
            str(payload["output_space"]),
        )
