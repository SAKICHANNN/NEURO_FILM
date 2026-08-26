"""Deterministic read-only render-recipe history for a future desktop surface."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .render_contract import RenderContractError, validate_render_recipe

RECIPE_HISTORY_SCHEMA_ID = "kmcfm.desktop-recipe-history.v1"
DEFAULT_MAXIMUM_RECIPE_FILES = 10_000
DEFAULT_MAXIMUM_RECIPE_BYTES = 2 * 1024 * 1024


class RecipeHistoryError(ValueError):
    """Raised when a recipe-history root or bounded scan is invalid."""


def _discover_recipe_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names[:] = sorted(
            name for name in directory_names if not (current_path / name).is_symlink()
        )
        for name in sorted(file_names):
            if name.endswith(".recipe.json"):
                paths.append(current_path / name)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _invalid_row(recipe_path: str, code: str) -> dict[str, Any]:
    return {
        "status": "invalid",
        "recipe_path": recipe_path,
        "error_code": code,
    }


def _enabled_effects(recipe: Mapping[str, Any]) -> list[str]:
    effects = recipe["render"]["effects"]
    enabled: list[str] = []
    for name in ("grain", "halation", "dust"):
        if float(effects[name]["strength"]) > 0.0:
            enabled.append(name)
    return enabled


def _valid_row(
    recipe_path: str, recipe_sha256: str, recipe: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": "valid",
        "recipe_path": recipe_path,
        "recipe_sha256": recipe_sha256,
        "profile_id": recipe["profile"]["profile_id"],
        "profile_version": recipe["profile"]["profile_version"],
        "input_path": recipe["input"]["path"],
        "input_sha256": recipe["input"]["sha256"],
        "input_color_state": recipe["input"]["color_state"],
        "style": recipe["render"]["style"],
        "seed": recipe["render"]["seed"],
        "enabled_effects": _enabled_effects(recipe),
        "output_path": recipe["output"]["path"],
        "output_sha256": recipe["output"]["sha256"],
        "output_format": recipe["output"]["format"],
        "output_bit_depth": recipe["output"]["bit_depth"],
        "output_label": recipe["claim"]["output_label"],
        "evidence_grade": recipe["claim"]["evidence_grade"],
        "software_commit": recipe["software"]["commit"],
    }


def _read_recipe_row(
    path: Path,
    *,
    root: Path,
    maximum_recipe_bytes: int,
) -> dict[str, Any]:
    recipe_path = path.relative_to(root).as_posix()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return _invalid_row(recipe_path, "recipe_unavailable")
    if not resolved.is_relative_to(root):
        return _invalid_row(recipe_path, "recipe_path_escape")
    if not resolved.is_file():
        return _invalid_row(recipe_path, "recipe_not_regular_file")
    try:
        with resolved.open("rb") as handle:
            payload = handle.read(maximum_recipe_bytes + 1)
    except OSError:
        return _invalid_row(recipe_path, "recipe_read_error")
    if len(payload) > maximum_recipe_bytes:
        return _invalid_row(recipe_path, "recipe_too_large")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        return _invalid_row(recipe_path, "recipe_invalid_utf8")
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError:
        return _invalid_row(recipe_path, "recipe_invalid_json")
    if not isinstance(value, Mapping):
        return _invalid_row(recipe_path, "recipe_not_object")
    try:
        validate_render_recipe(value)
    except (RenderContractError, KeyError, TypeError, ValueError):
        return _invalid_row(recipe_path, "recipe_contract_invalid")
    return _valid_row(recipe_path, hashlib.sha256(payload).hexdigest(), value)


def build_render_recipe_history(
    root: Path,
    *,
    maximum_recipe_files: int = DEFAULT_MAXIMUM_RECIPE_FILES,
    maximum_recipe_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES,
) -> dict[str, Any]:
    """Summarize strict recipes without reading their referenced input/output files."""
    if type(maximum_recipe_files) is not int or maximum_recipe_files <= 0:
        raise RecipeHistoryError("maximum_recipe_files must be a positive integer")
    if type(maximum_recipe_bytes) is not int or maximum_recipe_bytes <= 0:
        raise RecipeHistoryError("maximum_recipe_bytes must be a positive integer")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RecipeHistoryError("recipe-history root is unavailable") from exc
    if not resolved_root.is_dir():
        raise RecipeHistoryError("recipe-history root must be a directory")

    paths = _discover_recipe_paths(resolved_root)
    if len(paths) > maximum_recipe_files:
        raise RecipeHistoryError("recipe-history file limit exceeded")
    entries = [
        _read_recipe_row(
            path,
            root=resolved_root,
            maximum_recipe_bytes=maximum_recipe_bytes,
        )
        for path in paths
    ]
    valid_count = sum(row["status"] == "valid" for row in entries)
    invalid_count = len(entries) - valid_count
    if not entries:
        status = "empty"
    elif invalid_count == 0:
        status = "ready"
    elif valid_count == 0:
        status = "invalid"
    else:
        status = "partial"
    return {
        "schema_id": RECIPE_HISTORY_SCHEMA_ID,
        "status": status,
        "counts": {
            "discovered": len(entries),
            "valid": valid_count,
            "invalid": invalid_count,
        },
        "entries": entries,
    }


__all__ = [
    "DEFAULT_MAXIMUM_RECIPE_BYTES",
    "DEFAULT_MAXIMUM_RECIPE_FILES",
    "RECIPE_HISTORY_SCHEMA_ID",
    "RecipeHistoryError",
    "build_render_recipe_history",
]
