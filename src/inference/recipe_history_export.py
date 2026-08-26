"""Select one bounded recipe-history entry and replay it to a new file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .recipe_history import (
    DEFAULT_MAXIMUM_RECIPE_BYTES,
    DEFAULT_MAXIMUM_RECIPE_FILES,
    build_render_recipe_history,
)
from .render_contract import RenderContractError, validate_render_recipe
from .style_safe_engine import replay_style_safe_recipe_to_file


class RecipeHistoryExportError(ValueError):
    """Raised when a history selection cannot be exported exactly."""


def _validated_relative_recipe_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RecipeHistoryExportError("recipe_path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RecipeHistoryExportError("recipe_path must be a normalized relative path")
    if path.as_posix() != value or not value.endswith(".recipe.json"):
        raise RecipeHistoryExportError("recipe_path must name a normalized recipe file")
    return path


def export_recipe_history_entry(
    history_root: Path,
    *,
    recipe_path: str,
    profile_path: Path,
    output_path: Path,
    root: Path,
    maximum_recipe_files: int = DEFAULT_MAXIMUM_RECIPE_FILES,
    maximum_recipe_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES,
    tile_size: int | None = None,
) -> dict[str, Any]:
    """Replay one exact valid history row without reading its previous output."""

    relative = _validated_relative_recipe_path(recipe_path)
    if output_path.exists():
        raise RecipeHistoryExportError("export output path already exists")
    if not output_path.parent.is_dir():
        raise RecipeHistoryExportError("export output parent must already exist")

    catalog = build_render_recipe_history(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    selected = [row for row in catalog["entries"] if row["recipe_path"] == recipe_path]
    if len(selected) != 1 or selected[0]["status"] != "valid":
        raise RecipeHistoryExportError("recipe history selection is missing or invalid")
    row = selected[0]

    resolved_history_root = history_root.resolve(strict=True)
    selected_path = resolved_history_root.joinpath(*relative.parts).resolve(strict=True)
    if not selected_path.is_relative_to(resolved_history_root):
        raise RecipeHistoryExportError("recipe history selection escaped its root")
    payload = selected_path.read_bytes()
    if len(payload) > maximum_recipe_bytes:
        raise RecipeHistoryExportError("selected recipe exceeds the byte limit")
    recipe_sha256 = hashlib.sha256(payload).hexdigest()
    if recipe_sha256 != row["recipe_sha256"]:
        raise RecipeHistoryExportError("selected recipe changed after history validation")
    try:
        recipe = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecipeHistoryExportError("selected recipe cannot be decoded") from exc
    if not isinstance(recipe, Mapping):
        raise RecipeHistoryExportError("selected recipe is not an object")
    try:
        validate_render_recipe(recipe)
    except (RenderContractError, KeyError, TypeError, ValueError) as exc:
        raise RecipeHistoryExportError("selected recipe contract is invalid") from exc

    output_sha256 = replay_style_safe_recipe_to_file(
        recipe,
        profile_path=profile_path,
        output_path=output_path,
        root=root,
        tile_size=tile_size,
    )
    return {
        "recipe_path": recipe_path,
        "recipe_sha256": recipe_sha256,
        "style": row["style"],
        "output_path": str(output_path),
        "output_sha256": output_sha256,
    }


__all__ = ["RecipeHistoryExportError", "export_recipe_history_entry"]
