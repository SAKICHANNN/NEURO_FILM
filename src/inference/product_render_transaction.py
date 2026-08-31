"""Create-only transactions for private product image delivery."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    publish_create_only,
    remove_if_published,
)


class ProductRenderTransactionError(ValueError):
    """Reject an unsafe product primary-image destination."""


@dataclass(frozen=True)
class _StageSeal:
    identity: PublishedFileIdentity
    sha256: str
    size: int


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path.resolve(strict=False))))


def _reject_existing(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise ProductRenderTransactionError(
        f"product {label} destination must not already exist"
    )


def preflight_product_primary_output(input_path: Path, output_path: Path) -> None:
    """Reject an existing or input-alias destination without reading input pixels."""

    source = Path(input_path)
    destination = Path(output_path)
    if _normalized_path(source) == _normalized_path(destination):
        raise ProductRenderTransactionError(
            "product output must not identify the input path"
        )
    _reject_existing(destination, label="output")


@dataclass
class ProductImageRecipeTransaction:
    """Owned sibling stages and identity-safe publication for one pair."""

    image_path: Path
    recipe_path: Path
    image_stage: Path
    recipe_stage: Path
    _image_stage_seal: _StageSeal | None = None
    _recipe_stage_seal: _StageSeal | None = None
    _image_publication: PublishedFileIdentity | None = None
    _committed: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.cleanup()

    def stage_recipe(self, payload: Mapping[str, Any]) -> str:
        """Write canonical existing-recipe bytes to the owned unique stage."""

        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.recipe_stage.parent.mkdir(parents=True, exist_ok=True)
        with self.recipe_stage.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._recipe_stage_seal = _seal_stage(self.recipe_stage)
        return self._recipe_stage_seal.sha256

    def bind_image_stage(self) -> None:
        """Bind the just-encoded image stage before later publication."""

        if self._image_stage_seal is not None:
            raise ProductRenderTransactionError("product image stage already bound")
        self._image_stage_seal = _seal_stage(self.image_stage)

    def publish(self) -> None:
        """Publish image then recipe, rolling back only the owned image."""

        _require_seal(self._image_stage_seal, label="image stage")
        _require_seal(self._recipe_stage_seal, label="recipe stage")
        self._image_publication = publish_create_only(
            self.image_stage,
            self.image_path,
        )
        self._image_stage_seal = None
        try:
            publish_create_only(self.recipe_stage, self.recipe_path)
            self._recipe_stage_seal = None
        except BaseException:
            remove_if_published(self._image_publication)
            self._image_publication = None
            raise
        self._image_publication = None
        self._committed = True

    def cleanup(self) -> None:
        """Remove owned stages and an owned incomplete image, if any."""

        if self._image_stage_seal is not None:
            remove_if_published(self._image_stage_seal.identity)
            self._image_stage_seal = None
        if self._recipe_stage_seal is not None:
            remove_if_published(self._recipe_stage_seal.identity)
            self._recipe_stage_seal = None
        if self._image_publication is not None and not self._committed:
            remove_if_published(self._image_publication)
            self._image_publication = None


def _seal_stage(path: Path) -> _StageSeal:
    current = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ProductRenderTransactionError("product stage must be a regular file")
    return _StageSeal(
        identity=PublishedFileIdentity(
            path=path,
            device=current.st_dev,
            inode=current.st_ino,
        ),
        sha256=_sha256_file(path),
        size=current.st_size,
    )


def _require_seal(
    seal: _StageSeal | None,
    *,
    label: str,
) -> None:
    if seal is None:
        raise ProductRenderTransactionError(f"product {label} is not bound")
    try:
        current = seal.identity.path.lstat()
    except FileNotFoundError as exc:
        raise ProductRenderTransactionError(f"product {label} is missing") from exc
    if (current.st_dev, current.st_ino) != (
        seal.identity.device,
        seal.identity.inode,
    ):
        raise ProductRenderTransactionError(f"product {label} identity changed")
    if current.st_size != seal.size or _sha256_file(seal.identity.path) != seal.sha256:
        raise ProductRenderTransactionError(f"product {label} content changed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_product_image_recipe_transaction(
    input_path: Path,
    output_path: Path,
) -> ProductImageRecipeTransaction:
    """Preflight an absent distinct image/recipe pair before pixel decode."""

    source = Path(input_path)
    image = Path(output_path)
    recipe = image.with_suffix(".recipe.json")
    normalized = {
        _normalized_path(source),
        _normalized_path(image),
        _normalized_path(recipe),
    }
    if len(normalized) != 3:
        raise ProductRenderTransactionError(
            "product input, output and recipe paths must be distinct"
        )
    _reject_existing(image, label="output")
    _reject_existing(recipe, label="recipe")
    token = uuid.uuid4().hex
    image_stage = image.with_name(f".{image.stem}.{token}{image.suffix}")
    recipe_stage = recipe.with_name(f".{recipe.name}.{token}.stage")
    return ProductImageRecipeTransaction(
        image_path=image,
        recipe_path=recipe,
        image_stage=image_stage,
        recipe_stage=recipe_stage,
    )


__all__ = [
    "ProductImageRecipeTransaction",
    "ProductRenderTransactionError",
    "preflight_product_primary_output",
    "prepare_product_image_recipe_transaction",
]
