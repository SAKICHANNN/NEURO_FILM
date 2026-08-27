"""Authoritative evidence-bounded product look discovery and dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .generic_bw_look import list_generic_bw_looks, render_generic_bw_look_rgb
from .style_safe_engine import StyleSafeEngineError
from .three_stock_look import list_three_stock_looks, render_three_stock_look_rgb


def _build_catalog() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source in list_three_stock_looks():
        rows.append(
            {
                "look_id": source["style_id"],
                "display_name": source["display_name"],
                "film_stock_id": source["film_stock_id"],
                "process_family": source["process_family"],
                "look_family": "colour-stock-look-approximation",
                "evidence_tier": source["evidence_tier"],
                "claim_ceiling": source["claim_ceiling"],
            }
        )
    generic = list_generic_bw_looks()[0]
    rows.append(
        {
            "look_id": generic["look_id"],
            "display_name": generic["display_name"],
            "film_stock_id": None,
            "process_family": generic["process_family"],
            "look_family": "generic-bw-look-approximation",
            "evidence_tier": generic["evidence_tier"],
            "claim_ceiling": generic["claim_ceiling"],
        }
    )
    return tuple(rows)


PRODUCT_LOOK_CATALOG = _build_catalog()
_BY_LOOK_ID = {row["look_id"]: row for row in PRODUCT_LOOK_CATALOG}
_THREE_STOCK_BY_STYLE = {
    row["style_id"]: row for row in list_three_stock_looks()
}


def list_product_looks() -> tuple[dict[str, Any], ...]:
    """Return the full current product-look catalog as defensive copies."""

    return tuple(dict(row) for row in PRODUCT_LOOK_CATALOG)


def render_product_look_rgb(
    encoded_srgb: np.ndarray,
    *,
    profile: Mapping[str, Any],
    look_id: str,
    look_amount: float,
    style_statistics: Mapping[str, Mapping[str, Any]],
    guardrails: Mapping[str, Mapping[str, Any]],
    seed: int,
    tile_size: int | None = None,
    gamut_workers: int = 1,
    tile_workers: int = 1,
) -> np.ndarray:
    """Dispatch one manually selected product look to its frozen explicit core."""

    if not isinstance(look_id, str) or look_id not in _BY_LOOK_ID:
        raise StyleSafeEngineError("unsupported product look")
    if look_id == "generic_bw":
        execution_style = "hp5"
        if execution_style not in style_statistics or execution_style not in guardrails:
            raise StyleSafeEngineError("generic B&W runtime assets are incomplete")
        return render_generic_bw_look_rgb(
            encoded_srgb,
            profile=profile,
            look_amount=look_amount,
            style_statistics=style_statistics[execution_style],
            guardrails=guardrails[execution_style],
            seed=seed,
            tile_size=tile_size,
            gamut_workers=gamut_workers,
            tile_workers=tile_workers,
        )
    source = _THREE_STOCK_BY_STYLE[look_id]
    if look_id not in style_statistics or look_id not in guardrails:
        raise StyleSafeEngineError("colour product-look runtime assets are incomplete")
    return render_three_stock_look_rgb(
        encoded_srgb,
        profile=profile,
        film_stock_id=source["film_stock_id"],
        look_amount=look_amount,
        style_statistics=style_statistics[look_id],
        guardrails=guardrails[look_id],
        seed=seed,
        tile_size=tile_size,
        gamut_workers=gamut_workers,
        tile_workers=tile_workers,
    )


__all__ = ["PRODUCT_LOOK_CATALOG", "list_product_looks", "render_product_look_rgb"]
