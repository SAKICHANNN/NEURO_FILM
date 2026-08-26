"""Evidence-bounded user-selected three-stock look execution.

These entries are deterministic Look Approximation baselines.  The selected
stock name is a user intent label, not a claim that the operator is a measured
or calibrated stock response.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from src.color_engine.safe_lab_rgb_context import build_safe_lab_source_context

from .render_contract import validate_render_profile
from .style_safe_engine import StyleSafeEngineError, render_resolved_safe_lab_rgb

THREE_STOCK_LOOK_CATALOG: tuple[dict[str, str], ...] = (
    {
        "film_stock_id": "fujifilm_velvia_50",
        "style_id": "velvia_50",
        "display_name": "Fujifilm Velvia 50",
        "process_family": "E-6 slide",
        "evidence_tier": "display-proxy-look-approximation",
        "claim_ceiling": "Velvia 50 display-proxy Look Approximation baseline; not a calibrated stock response",
    },
    {
        "film_stock_id": "kodak_portra_400",
        "style_id": "portra_400",
        "display_name": "Kodak Portra 400",
        "process_family": "C-41 colour negative",
        "evidence_tier": "legacy-unpaired-look-approximation",
        "claim_ceiling": "legacy unpaired Portra 400 Look Approximation baseline; controlled stock evidence remains a data gap",
    },
    {
        "film_stock_id": "kodak_ektar_100",
        "style_id": "ektar_100",
        "display_name": "Kodak Ektar 100",
        "process_family": "C-41 colour negative",
        "evidence_tier": "legacy-unpaired-look-approximation",
        "claim_ceiling": "legacy unpaired Ektar 100 Look Approximation baseline; controlled stock evidence remains a data gap",
    },
)

_BY_STOCK = {row["film_stock_id"]: row for row in THREE_STOCK_LOOK_CATALOG}


def list_three_stock_looks() -> tuple[dict[str, str], ...]:
    """Return an isolated, stable discovery payload for the three baselines."""

    return tuple(dict(row) for row in THREE_STOCK_LOOK_CATALOG)


def _amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StyleSafeEngineError("look_amount must be a finite number in [0, 1]")
    amount = float(value)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise StyleSafeEngineError("look_amount must be a finite number in [0, 1]")
    return amount


def resolve_three_stock_look_parameters(
    profile: Mapping[str, Any], *, film_stock_id: str, look_amount: float
) -> tuple[str, dict[str, Any]]:
    """Resolve one bounded amount without changing the frozen profile asset."""

    validate_render_profile(profile)
    if film_stock_id not in _BY_STOCK:
        raise StyleSafeEngineError("unsupported three-stock look")
    amount = _amount(look_amount)
    style = _BY_STOCK[film_stock_id]["style_id"]
    if style not in profile["style_parameters"]:
        raise StyleSafeEngineError("three-stock style is absent from profile")
    parameters = dict(profile["style_parameters"][style])
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
    return style, parameters


def render_three_stock_look_rgb(
    encoded_srgb: np.ndarray,
    *,
    profile: Mapping[str, Any],
    film_stock_id: str,
    look_amount: float,
    style_statistics: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
    tile_size: int | None = None,
    gamut_workers: int = 1,
    tile_workers: int = 1,
) -> np.ndarray:
    """Render one manually selected bounded three-stock Look Approximation."""

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
    style, parameters = resolve_three_stock_look_parameters(
        profile, film_stock_id=film_stock_id, look_amount=amount
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


def iter_three_stock_look_rgb_shared_context(
    encoded_srgb: np.ndarray,
    *,
    profile: Mapping[str, Any],
    look_amount: float,
    style_statistics: Mapping[str, Mapping[str, Any]],
    guardrails: Mapping[str, Mapping[str, Any]],
    seed: int,
    tile_size: int,
    gamut_workers: int = 1,
    tile_workers: int = 1,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Yield all three tiled looks while computing source statistics once."""

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
    source_context = None if amount == 0.0 else build_safe_lab_source_context(source)
    for catalog_row in THREE_STOCK_LOOK_CATALOG:
        style, parameters = resolve_three_stock_look_parameters(
            profile,
            film_stock_id=catalog_row["film_stock_id"],
            look_amount=amount,
        )
        if style not in style_statistics or style not in guardrails:
            raise StyleSafeEngineError(f"missing three-stock inputs for style: {style}")
        if amount == 0.0:
            output = np.ascontiguousarray(source.copy())
        else:
            output = render_resolved_safe_lab_rgb(
                source,
                style=style,
                style_statistics=style_statistics[style],
                style_parameters=parameters,
                guardrails=guardrails[style],
                seed=seed,
                tile_size=tile_size,
                gamut_workers=gamut_workers,
                tile_workers=tile_workers,
                source_context=source_context,
            )
        yield dict(catalog_row), output


__all__ = [
    "THREE_STOCK_LOOK_CATALOG",
    "iter_three_stock_look_rgb_shared_context",
    "list_three_stock_looks",
    "render_three_stock_look_rgb",
    "resolve_three_stock_look_parameters",
]
