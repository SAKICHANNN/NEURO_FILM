"""Canonical L0 exposure/white-balance operators and roll nuisance gauge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import _validate_rgb


PHOTOMETRIC_SCHEMA = "roll2film.photometric_l0.v1"
GAUGE_SCHEMA = "roll2film.roll_nuisance_gauge.v1"
_GAUGE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PhotometricColorOperator:
    """Invertible L0 log-exposure plus unit-geometric-mean white balance."""

    log_exposure: float
    log_white_balance: np.ndarray
    working_space: str = "linear_srgb"

    def __post_init__(self) -> None:
        exposure = float(self.log_exposure)
        white_balance = np.asarray(self.log_white_balance, dtype=np.float64)
        if not np.isfinite(exposure):
            raise ValueError("log exposure must be finite")
        if white_balance.shape != (3,) or not np.all(np.isfinite(white_balance)):
            raise ValueError("log white balance must be three finite values")
        if abs(float(np.sum(white_balance))) > _GAUGE_TOLERANCE:
            raise ValueError("log white balance must sum to zero")
        if not isinstance(self.working_space, str) or not self.working_space:
            raise ValueError("working space must be a non-empty string")
        object.__setattr__(self, "log_exposure", exposure)
        object.__setattr__(self, "log_white_balance", white_balance)

    @classmethod
    def identity(cls, working_space: str = "linear_srgb") -> "PhotometricColorOperator":
        return cls(0.0, np.zeros(3, dtype=np.float64), working_space)

    @property
    def log_gains(self) -> np.ndarray:
        return self.log_exposure + self.log_white_balance

    @property
    def gains(self) -> np.ndarray:
        return np.exp(self.log_gains)

    @property
    def jacobian_determinant(self) -> float:
        return float(np.prod(self.gains))

    def jacobian(self) -> np.ndarray:
        return np.diag(self.gains)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return _validate_rgb(rgb) * self.gains

    def inverse(self, rgb: np.ndarray) -> np.ndarray:
        return _validate_rgb(rgb) / self.gains

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PHOTOMETRIC_SCHEMA,
            "working_space": self.working_space,
            "log_exposure": self.log_exposure,
            "log_white_balance": self.log_white_balance.tolist(),
            "jacobian_determinant": self.jacobian_determinant,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhotometricColorOperator":
        if payload.get("schema") != PHOTOMETRIC_SCHEMA:
            raise ValueError(f"unsupported photometric schema: {payload.get('schema')!r}")
        return cls(
            log_exposure=float(payload["log_exposure"]),
            log_white_balance=np.asarray(payload["log_white_balance"], dtype=np.float64),
            working_space=str(payload["working_space"]),
        )


@dataclass(frozen=True)
class RollNuisanceGauge:
    """Gauge-fixed per-frame nuisance and the exposure moved to the base look."""

    shared_log_exposure: float
    frame_log_exposure: np.ndarray
    frame_log_white_balance: np.ndarray

    def __post_init__(self) -> None:
        shared = float(self.shared_log_exposure)
        exposure = np.asarray(self.frame_log_exposure, dtype=np.float64)
        white_balance = np.asarray(self.frame_log_white_balance, dtype=np.float64)
        if exposure.ndim != 1 or len(exposure) == 0:
            raise ValueError("gauge requires at least one frame exposure")
        if white_balance.shape != (len(exposure), 3):
            raise ValueError("gauge white balance must have shape (frames, 3)")
        if not np.isfinite(shared) or not np.all(np.isfinite(exposure)) or not np.all(
            np.isfinite(white_balance)
        ):
            raise ValueError("gauge values must be finite")
        if abs(float(np.mean(exposure))) > _GAUGE_TOLERANCE:
            raise ValueError("gauge frame log exposure must have zero mean")
        if float(np.max(np.abs(np.sum(white_balance, axis=1)))) > _GAUGE_TOLERANCE:
            raise ValueError("every gauge log white balance row must sum to zero")
        object.__setattr__(self, "shared_log_exposure", shared)
        object.__setattr__(self, "frame_log_exposure", exposure)
        object.__setattr__(self, "frame_log_white_balance", white_balance)

    @property
    def recomposed_log_gains(self) -> np.ndarray:
        return (
            self.shared_log_exposure
            + self.frame_log_exposure[:, None]
            + self.frame_log_white_balance
        )

    def frame_operator(
        self, index: int, working_space: str = "linear_srgb"
    ) -> PhotometricColorOperator:
        if index < 0 or index >= len(self.frame_log_exposure):
            raise IndexError("frame index outside gauge")
        return PhotometricColorOperator(
            float(self.frame_log_exposure[index]),
            self.frame_log_white_balance[index],
            working_space,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GAUGE_SCHEMA,
            "shared_log_exposure": self.shared_log_exposure,
            "frame_log_exposure": self.frame_log_exposure.tolist(),
            "frame_log_white_balance": self.frame_log_white_balance.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RollNuisanceGauge":
        if payload.get("schema") != GAUGE_SCHEMA:
            raise ValueError(f"unsupported gauge schema: {payload.get('schema')!r}")
        return cls(
            float(payload["shared_log_exposure"]),
            np.asarray(payload["frame_log_exposure"], dtype=np.float64),
            np.asarray(payload["frame_log_white_balance"], dtype=np.float64),
        )


def canonicalize_roll_nuisance(
    frame_log_exposure: np.ndarray, frame_log_white_balance: np.ndarray
) -> RollNuisanceGauge:
    """Move WB scale and roll-average exposure into one explicit base offset."""

    exposure = np.asarray(frame_log_exposure, dtype=np.float64)
    white_balance = np.asarray(frame_log_white_balance, dtype=np.float64)
    if exposure.ndim != 1 or len(exposure) == 0:
        raise ValueError("frame log exposure must be a non-empty vector")
    if white_balance.shape != (len(exposure), 3):
        raise ValueError("frame log white balance must have shape (frames, 3)")
    if not np.all(np.isfinite(exposure)) or not np.all(np.isfinite(white_balance)):
        raise ValueError("raw nuisance values must be finite")
    row_scale = np.mean(white_balance, axis=1)
    centred_white_balance = white_balance - row_scale[:, None]
    exposure_with_wb_scale = exposure + row_scale
    shared = float(np.mean(exposure_with_wb_scale))
    centred_exposure = exposure_with_wb_scale - shared
    return RollNuisanceGauge(shared, centred_exposure, centred_white_balance)

