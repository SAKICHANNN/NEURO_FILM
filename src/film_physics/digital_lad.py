"""Typed Kodak H-387 Digital LAD code and printing-density primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

DIGITAL_LAD_AIM_SCHEMA = "neuro_film.film_physics.digital_lad_aim.v1"


class CineonRecordingMode(str, Enum):
    NEGATIVE = "negative"
    INTERPOSITIVE = "interpositive"


def _mode(value: CineonRecordingMode | str) -> CineonRecordingMode:
    try:
        return CineonRecordingMode(value)
    except ValueError as exc:
        raise ValueError("mode must be negative or interpositive") from exc


def _codes(value: np.ndarray | list[int] | int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("Cineon code values must be integers")
    codes = raw.astype(np.int64, copy=False)
    if np.any(codes < 0) or np.any(codes > 1023):
        raise ValueError("Cineon code values must be in [0, 1023]")
    return codes


def _readonly_float64(value: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DigitalLadDensityResult:
    mode: CineonRecordingMode
    code: np.ndarray
    raw_printing_density: np.ndarray
    physical_nonnegative_printing_density: np.ndarray

    def __post_init__(self) -> None:
        mode = _mode(self.mode)
        code = _codes(self.code).copy()
        raw = np.asarray(self.raw_printing_density, dtype=np.float64)
        physical = np.asarray(
            self.physical_nonnegative_printing_density, dtype=np.float64
        )
        if raw.shape != code.shape or physical.shape != code.shape:
            raise ValueError("density result arrays must match the code shape")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(physical)):
            raise ValueError("density result arrays must be finite")
        if np.any(physical < 0.0) or not np.array_equal(
            physical, np.maximum(raw, 0.0)
        ):
            raise ValueError("physical density must be the exact nonnegative raw density")
        code.setflags(write=False)
        raw = raw.copy()
        raw.setflags(write=False)
        physical = physical.copy()
        physical.setflags(write=False)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "raw_printing_density", raw)
        object.__setattr__(self, "physical_nonnegative_printing_density", physical)


def code_to_printing_density(
    code: np.ndarray | list[int] | int,
    mode: CineonRecordingMode | str,
) -> DigitalLadDensityResult:
    """Map 10-bit recorder code to documented raw and physical density domains."""

    resolved_mode = _mode(mode)
    codes = _codes(code)
    values = codes.astype(np.float64)
    if resolved_mode is CineonRecordingMode.NEGATIVE:
        raw = 0.002 * values
    else:
        raw = 1.930 - 0.002 * values
    return DigitalLadDensityResult(
        mode=resolved_mode,
        code=codes,
        raw_printing_density=raw,
        physical_nonnegative_printing_density=np.maximum(raw, 0.0),
    )


def raw_printing_density_to_code(
    density: np.ndarray | list[float] | float,
    mode: CineonRecordingMode | str,
) -> np.ndarray:
    """Invert the raw equation without silently clamping the IP negative tail."""

    resolved_mode = _mode(mode)
    values = np.asarray(density, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("raw printing density must be finite")
    if resolved_mode is CineonRecordingMode.NEGATIVE:
        if np.any(values < 0.0) or np.any(values > 0.002 * 1023):
            raise ValueError("negative-mode raw printing density is outside [0, 2.046]")
        return values / 0.002
    if np.any(values < 1.930 - 0.002 * 1023) or np.any(values > 1.930):
        raise ValueError("IP-mode raw printing density is outside [-0.116, 1.930]")
    return (1.930 - values) / 0.002


@dataclass(frozen=True)
class DigitalLadAim:
    stock_id: str
    mode: CineonRecordingMode
    code: int
    printing_density: float
    status_m_above_dmin: np.ndarray
    dmin: np.ndarray
    status_m_total: np.ndarray

    def __post_init__(self) -> None:
        if not self.stock_id:
            raise ValueError("stock_id must be non-empty")
        mode = _mode(self.mode)
        code = int(self.code)
        if isinstance(self.code, bool) or code != self.code or not 0 <= code <= 1023:
            raise ValueError("LAD code must be an integer in [0, 1023]")
        density = float(self.printing_density)
        if not np.isfinite(density):
            raise ValueError("printing_density must be finite")
        above = _readonly_float64(self.status_m_above_dmin, "status_m_above_dmin")
        dmin = _readonly_float64(self.dmin, "dmin")
        total = _readonly_float64(self.status_m_total, "status_m_total")
        if np.max(np.abs((above + dmin) - total)) > 1e-12:
            raise ValueError("Status M total must equal above-D-min plus D-min")
        expected = code_to_printing_density(code, mode).raw_printing_density
        if abs(float(expected) - density) > 1e-12:
            raise ValueError("LAD printing density disagrees with the mode equation")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "printing_density", density)
        object.__setattr__(self, "status_m_above_dmin", above)
        object.__setattr__(self, "dmin", dmin)
        object.__setattr__(self, "status_m_total", total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DIGITAL_LAD_AIM_SCHEMA,
            "stock_id": self.stock_id,
            "mode": self.mode.value,
            "code": self.code,
            "printing_density": self.printing_density,
            "status_m_above_dmin": self.status_m_above_dmin.tolist(),
            "dmin": self.dmin.tolist(),
            "status_m_total": self.status_m_total.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DigitalLadAim:
        if payload.get("schema") != DIGITAL_LAD_AIM_SCHEMA:
            raise ValueError("unsupported Digital LAD aim schema")
        return cls(
            stock_id=str(payload["stock_id"]),
            mode=_mode(str(payload["mode"])),
            code=payload["code"],
            printing_density=float(payload["printing_density"]),
            status_m_above_dmin=np.asarray(
                payload["status_m_above_dmin"], dtype=np.float64
            ),
            dmin=np.asarray(payload["dmin"], dtype=np.float64),
            status_m_total=np.asarray(payload["status_m_total"], dtype=np.float64),
        )


def aim_from_config(payload: dict[str, Any]) -> DigitalLadAim:
    return DigitalLadAim(
        stock_id=str(payload["stock_id"]),
        mode=_mode(str(payload["mode"])),
        code=payload["code"],
        printing_density=float(payload["printing_density"]),
        status_m_above_dmin=np.asarray(payload["status_m_above_dmin"], dtype=np.float64),
        dmin=np.asarray(payload["dmin"], dtype=np.float64),
        status_m_total=np.asarray(payload["status_m_total"], dtype=np.float64),
    )
