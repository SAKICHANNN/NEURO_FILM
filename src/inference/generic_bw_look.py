"""One evidence-bounded generic black-and-white Look Approximation.

The historical ``hp5`` and ``tri_x_400`` profile entries remain immutable for
recipe replay.  BW2.D0 did not distinguish them mechanically, so current
product discovery exposes only this generic look.  ``hp5`` is the frozen
execution identity selected by candidate order, not a named-stock claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .render_contract import validate_render_profile
from .style_safe_engine import StyleSafeEngineError, render_resolved_safe_lab_rgb

GENERIC_BW_LOOK_ID = "generic_bw"
_LEGACY_EXECUTION_STYLE_ID = "hp5"

GENERIC_BW_LOOK_CATALOG: tuple[dict[str, str], ...] = (
    {
        "look_id": GENERIC_BW_LOOK_ID,
        "display_name": "Classic Black & White",
        "film_stock_id": "generic_black_and_white",
        "process_family": "unspecified black-and-white negative",
        "evidence_tier": "generic-bw-look-approximation",
        "claim_ceiling": (
            "generic black-and-white Look Approximation; not an HP5, Tri-X, "
            "developer, process, calibrated stock-response or authenticity claim"
        ),
    },
)


def list_generic_bw_looks() -> tuple[dict[str, str], ...]:
    """Return the sole current B&W product look as defensive copies."""

    return tuple(dict(row) for row in GENERIC_BW_LOOK_CATALOG)


def _amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StyleSafeEngineError("look_amount must be a finite number in [0, 1]")
    amount = float(value)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise StyleSafeEngineError("look_amount must be a finite number in [0, 1]")
    return amount


def resolve_generic_bw_look_parameters(
    profile: Mapping[str, Any], *, look_amount: float
) -> tuple[str, dict[str, Any]]:
    """Resolve the generic look without mutating the frozen legacy profile."""

    validate_render_profile(profile)
    amount = _amount(look_amount)
    if _LEGACY_EXECUTION_STYLE_ID not in profile["style_parameters"]:
        raise StyleSafeEngineError("generic B&W execution style is absent from profile")
    parameters = dict(profile["style_parameters"][_LEGACY_EXECUTION_STYLE_ID])
    for key in (
        "strength",
        "luma_strength",
        "grain",
        "chroma_curve_strength",
        "tone_rolloff",
        "dither",
    ):
        parameters[key] = float(parameters[key]) * amount
    parameters["shadow_floor_l"] = float(parameters["shadow_floor_l"]) * amount
    parameters["highlight_ceiling_l"] = (
        100.0 - (100.0 - float(parameters["highlight_ceiling_l"])) * amount
    )
    if amount == 0.0:
        parameters["output_margin"] = 0
    return _LEGACY_EXECUTION_STYLE_ID, parameters


def render_generic_bw_look_rgb(
    encoded_srgb: np.ndarray,
    *,
    profile: Mapping[str, Any],
    look_amount: float,
    style_statistics: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
    tile_size: int | None = None,
    gamut_workers: int = 1,
    tile_workers: int = 1,
) -> np.ndarray:
    """Render the generic B&W look through its frozen legacy execution identity."""

    source = np.asarray(encoded_srgb)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or source.size == 0
        or not np.isfinite(source).all()
        or np.any((source < 0.0) | (source > 1.0))
    ):
        raise StyleSafeEngineError("encoded_srgb must be finite bounded HxWx3 float32")
    amount = _amount(look_amount)
    style, parameters = resolve_generic_bw_look_parameters(
        profile, look_amount=amount
    )
    if amount == 0.0:
        return np.ascontiguousarray(source.copy())
    return render_resolved_safe_lab_rgb(
        source,
        style=style,
        style_statistics=style_statistics,
        style_parameters=parameters,
        guardrails=guardrails,
        seed=seed,
        tile_size=tile_size,
        gamut_workers=gamut_workers,
        tile_workers=tile_workers,
    )


__all__ = [
    "GENERIC_BW_LOOK_CATALOG",
    "GENERIC_BW_LOOK_ID",
    "list_generic_bw_looks",
    "render_generic_bw_look_rgb",
    "resolve_generic_bw_look_parameters",
]
