"""Typed neutral Digital LAD anchors for an existing print interpretation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.roll2film.sensitometry_print import DensityToPrintInterpretation

from .digital_lad import (
    CineonRecordingMode,
    DigitalLadAim,
    code_to_printing_density,
)
from .exposure_development import InterpretationRoute

DIGITAL_LAD_PRINT_ANCHOR_SCHEMA = (
    "neuro_film.film_physics.digital_lad_print_anchor.v1"
)


@dataclass(frozen=True)
class DigitalLadPrintAnchorResult:
    mode: CineonRecordingMode
    code: np.ndarray
    raw_printing_density: np.ndarray
    neutral_print_reflectance: np.ndarray
    print_operator_identity: str

    def __post_init__(self) -> None:
        mode = CineonRecordingMode(self.mode)
        code = np.asarray(self.code)
        density = np.asarray(self.raw_printing_density, dtype=np.float64)
        reflectance = np.asarray(self.neutral_print_reflectance, dtype=np.float64)
        if code.dtype.kind not in "iu" or code.dtype.kind == "b":
            raise ValueError("anchor code must be integral")
        if density.shape != code.shape or reflectance.shape != (*code.shape, 3):
            raise ValueError("anchor result shapes do not match")
        if (
            np.any(code < 0)
            or np.any(code > 1023)
            or not np.all(np.isfinite(density))
            or np.any(density < 0.0)
            or not np.all(np.isfinite(reflectance))
            or np.any(reflectance < 0.0)
            or np.any(reflectance > 1.0)
        ):
            raise ValueError("anchor result left its typed domain")
        if not isinstance(self.print_operator_identity, str) or not self.print_operator_identity:
            raise ValueError("print operator identity must be non-empty")
        owned_code = np.array(code, dtype=np.int64, copy=True, order="C")
        owned_density = np.array(density, dtype=np.float64, copy=True, order="C")
        owned_reflectance = np.array(
            reflectance, dtype=np.float64, copy=True, order="C"
        )
        for value in (owned_code, owned_density, owned_reflectance):
            value.setflags(write=False)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "code", owned_code)
        object.__setattr__(self, "raw_printing_density", owned_density)
        object.__setattr__(self, "neutral_print_reflectance", owned_reflectance)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DIGITAL_LAD_PRINT_ANCHOR_SCHEMA,
            "role": "recorder_printing_density_neutral_anchor",
            "mode": self.mode.value,
            "code_shape": list(self.code.shape),
            "channel_order": ["red", "green", "blue"],
            "accepted_route": InterpretationRoute.COLOR_NEGATIVE_PRINT.value,
            "print_operator_identity": self.print_operator_identity,
            "status_m_used_in_arithmetic": False,
            "dmin_used_in_arithmetic": False,
            "developed_exposure_result_fabricated": False,
        }


def _print_identity(operator: DensityToPrintInterpretation) -> str:
    encoded = json.dumps(
        operator.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_digital_lad_codes_to_print(
    code: np.ndarray | list[int] | int,
    *,
    mode: CineonRecordingMode | str,
    route: InterpretationRoute | str,
    print_interpretation: DensityToPrintInterpretation,
) -> DigitalLadPrintAnchorResult:
    """Apply nonnegative neutral recorder density without changing its role."""
    if InterpretationRoute(route) is not InterpretationRoute.COLOR_NEGATIVE_PRINT:
        raise ValueError("Digital LAD neutral anchor requires color_negative_print")
    if not isinstance(print_interpretation, DensityToPrintInterpretation):
        raise TypeError("print_interpretation must be DensityToPrintInterpretation")
    resolved = code_to_printing_density(code, mode)
    if np.any(resolved.raw_printing_density < 0.0):
        raise ValueError("negative raw printing density cannot enter print interpretation")
    neutral = np.repeat(resolved.raw_printing_density[..., None], 3, axis=-1)
    reflectance = print_interpretation.apply(neutral.reshape(-1, 3)).reshape(
        (*resolved.code.shape, 3)
    )
    return DigitalLadPrintAnchorResult(
        resolved.mode,
        resolved.code,
        resolved.raw_printing_density,
        reflectance,
        _print_identity(print_interpretation),
    )


def apply_digital_lad_aim_to_print(
    aim: DigitalLadAim,
    *,
    route: InterpretationRoute | str,
    print_interpretation: DensityToPrintInterpretation,
) -> DigitalLadPrintAnchorResult:
    """Apply one H-387 aim while leaving Status M and D-min out of arithmetic."""
    if not isinstance(aim, DigitalLadAim):
        raise TypeError("aim must be DigitalLadAim")
    result = apply_digital_lad_codes_to_print(
        aim.code,
        mode=aim.mode,
        route=route,
        print_interpretation=print_interpretation,
    )
    if abs(result.raw_printing_density.item() - aim.printing_density) > 1e-12:
        raise ValueError("aim printing density disagrees with the typed equation")
    return result


__all__ = [
    "DIGITAL_LAD_PRINT_ANCHOR_SCHEMA",
    "DigitalLadPrintAnchorResult",
    "apply_digital_lad_aim_to_print",
    "apply_digital_lad_codes_to_print",
]
