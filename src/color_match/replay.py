"""Atomic persistence and deterministic replay for reference-look recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.inference.render_contract import atomic_write_json, sha256_file
from src.preprocess.types import WorkingImage

from .contracts import (
    ReferenceLookRecipe,
    ReferenceMatchContractError,
    recipe_from_json,
    validate_recipe,
)
from .render import ReferenceMatchResult, render_reference_batch
from .safety import (
    GuardedReferenceMatchResult,
    render_reference_batch_guarded,
)


_MAX_RECIPE_BYTES = 1024 * 1024


def save_reference_look_recipe(
    recipe: ReferenceLookRecipe,
    path: Path | str,
) -> str:
    """Atomically save one validated recipe and return its file SHA-256."""

    validate_recipe(recipe)
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise ReferenceMatchContractError("recipe path must not be a directory")
    atomic_write_json(destination, recipe.to_dict())
    return sha256_file(destination)


def load_reference_look_recipe(path: Path | str) -> ReferenceLookRecipe:
    """Load one bounded UTF-8 recipe file and fail closed on any mismatch."""

    source = Path(path)
    if not source.is_file():
        raise ReferenceMatchContractError("recipe path must be an existing file")
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ReferenceMatchContractError("recipe file metadata is unreadable") from exc
    if size <= 0 or size > _MAX_RECIPE_BYTES:
        raise ReferenceMatchContractError(
            "recipe file size must be within the bounded contract"
        )
    try:
        encoded = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReferenceMatchContractError("recipe file must be readable UTF-8") from exc
    return recipe_from_json(encoded)


def replay_reference_batch(
    recipe_path: Path | str,
    sources: Iterable[WorkingImage],
) -> tuple[ReferenceMatchResult, ...]:
    """Load a frozen recipe and apply it to a non-empty ordered source batch."""

    return render_reference_batch(load_reference_look_recipe(recipe_path), sources)


def replay_reference_batch_guarded(
    recipe_path: Path | str,
    sources: Iterable[WorkingImage],
) -> tuple[GuardedReferenceMatchResult, ...]:
    """Load a frozen recipe and reproduce the default product guard."""

    return render_reference_batch_guarded(
        load_reference_look_recipe(recipe_path),
        sources,
    )


__all__ = [
    "load_reference_look_recipe",
    "replay_reference_batch",
    "replay_reference_batch_guarded",
    "save_reference_look_recipe",
]
