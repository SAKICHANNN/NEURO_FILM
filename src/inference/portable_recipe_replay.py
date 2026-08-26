"""Caller-driven replay from one validated portable recipe bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .portable_recipe_bundle import bind_portable_recipe_recovery_bundle
from .recipe_recovery_bundle import _strict_json
from .style_safe_engine import replay_style_safe_recipe_to_file


def replay_portable_recipe_recovery_bundle_to_file(
    *,
    bundle_path: Path,
    input_path: Path,
    output_path: Path,
    recipe_path: Path,
    profile_path: Path,
    root: Path,
    tile_size: int | None = None,
) -> dict[str, Any]:
    """Bind one portable bundle and replay its exact strict recipe once."""

    binding = bind_portable_recipe_recovery_bundle(
        bundle_path=bundle_path,
        input_path=input_path,
        output_path=output_path,
        recipe_path=recipe_path,
    )
    recipe = _strict_json(recipe_path.read_bytes(), "bound replay recipe")
    if recipe["input"]["path"] != str(input_path):
        raise ValueError("bound replay input path drifted")
    if recipe["output"]["path"] != str(output_path):
        raise ValueError("bound replay output path drifted")
    output_sha256 = replay_style_safe_recipe_to_file(
        recipe,
        profile_path=profile_path,
        output_path=output_path,
        root=root,
        tile_size=tile_size,
    )
    return {
        "schema_id": "kmcfm.portable-recipe-replay-receipt.v1",
        "bundle_sha256": binding["bundle_sha256"],
        "bound_recipe_sha256": binding["recipe_sha256"],
        "source_recipe_semantic_sha256": binding[
            "source_recipe_semantic_sha256"
        ],
        "input_sha256": binding["input_sha256"],
        "output_sha256": output_sha256,
        "style": binding["style"],
        "claim": binding["claim"],
        "caller_driven": True,
    }


__all__ = ["replay_portable_recipe_recovery_bundle_to_file"]
