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


@dataclass(frozen=True)
class _DirectoryStageSeal:
    identity: PublishedFileIdentity
    files: tuple[tuple[str, _StageSeal], ...]


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


@dataclass
class ProductRenderBundleTransaction:
    """Owned stages and identity-safe publication for requested product artifacts."""

    image_path: Path
    image_stage: Path
    recipe_path: Path | None = None
    recipe_stage: Path | None = None
    layer_path: Path | None = None
    layer_stage: Path | None = None
    metrics_path: Path | None = None
    metrics_stage: Path | None = None
    _image_stage_seal: _StageSeal | None = None
    _recipe_stage_seal: _StageSeal | None = None
    _layer_stage_seal: _DirectoryStageSeal | None = None
    _metrics_stage_seal: _StageSeal | None = None
    _published_files: list[PublishedFileIdentity] | None = None
    _published_layer_root: PublishedFileIdentity | None = None
    _committed: bool = False

    def __enter__(self) -> Self:
        self._published_files = []
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.cleanup()

    def bind_image_stage(self) -> None:
        if self._image_stage_seal is not None:
            raise ProductRenderTransactionError("product image stage already bound")
        self._image_stage_seal = _seal_stage(self.image_stage)

    def stage_recipe(self, payload: Mapping[str, Any]) -> str:
        if self.recipe_stage is None:
            raise ProductRenderTransactionError("product recipe was not requested")
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self._recipe_stage_seal = _write_stage(self.recipe_stage, encoded)
        return self._recipe_stage_seal.sha256

    def bind_layer_stage(self) -> None:
        if self.layer_stage is None:
            raise ProductRenderTransactionError("product layers were not requested")
        _require_directory_seal(self._layer_stage_seal, label="layer stage")
        assert self._layer_stage_seal is not None
        if not self._layer_stage_seal.files:
            raise ProductRenderTransactionError("product layer stage must not be empty")

    def prepare_layer_stage(self) -> None:
        if self.layer_stage is None:
            raise ProductRenderTransactionError("product layers were not requested")
        if self._layer_stage_seal is not None:
            raise ProductRenderTransactionError("product layer stage already prepared")
        self.layer_stage.mkdir()
        current = self.layer_stage.lstat()
        self._layer_stage_seal = _DirectoryStageSeal(
            identity=PublishedFileIdentity(
                path=self.layer_stage,
                device=current.st_dev,
                inode=current.st_ino,
            ),
            files=(),
        )

    def bind_layer_file(self, path: Path) -> None:
        if self.layer_stage is None or self._layer_stage_seal is None:
            raise ProductRenderTransactionError("product layer stage is not prepared")
        candidate = Path(path)
        if candidate.parent != self.layer_stage or candidate.name in {
            relative for relative, _seal in self._layer_stage_seal.files
        }:
            raise ProductRenderTransactionError("invalid product layer stage file")
        file_seal = _seal_stage(candidate)
        self._layer_stage_seal = _DirectoryStageSeal(
            identity=self._layer_stage_seal.identity,
            files=(*self._layer_stage_seal.files, (candidate.name, file_seal)),
        )

    def stage_metrics(self, payload: Mapping[str, Any]) -> str:
        if self.metrics_stage is None:
            raise ProductRenderTransactionError("product metrics were not requested")
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self._metrics_stage_seal = _write_stage(self.metrics_stage, encoded)
        return self._metrics_stage_seal.sha256

    def publish(self) -> None:
        """Publish the complete requested bundle or roll back matching owned entries."""

        _require_seal(self._image_stage_seal, label="image stage")
        if self.recipe_stage is not None:
            _require_seal(self._recipe_stage_seal, label="recipe stage")
        if self.layer_stage is not None:
            _require_directory_seal(self._layer_stage_seal, label="layer stage")
        if self.metrics_stage is not None:
            _require_seal(self._metrics_stage_seal, label="metrics stage")
        if self._published_files is None:
            self._published_files = []
        try:
            self._publish_file(self.image_stage, self.image_path)
            self._image_stage_seal = None
            if self.recipe_stage is not None and self.recipe_path is not None:
                self._publish_file(self.recipe_stage, self.recipe_path)
                self._recipe_stage_seal = None
            if self.layer_stage is not None and self.layer_path is not None:
                self._publish_layers()
            if self.metrics_stage is not None and self.metrics_path is not None:
                self._publish_file(self.metrics_stage, self.metrics_path)
                self._metrics_stage_seal = None
        except BaseException:
            self._rollback_publications()
            raise
        self._published_files.clear()
        self._published_layer_root = None
        self._committed = True

    def cleanup(self) -> None:
        if not self._committed:
            self._rollback_publications()
        self._remove_stage(self._image_stage_seal)
        self._image_stage_seal = None
        self._remove_stage(self._recipe_stage_seal)
        self._recipe_stage_seal = None
        self._remove_stage(self._metrics_stage_seal)
        self._metrics_stage_seal = None
        self._remove_layer_stage()

    def _publish_file(self, stage: Path, final: Path) -> None:
        assert self._published_files is not None
        self._published_files.append(publish_create_only(stage, final))

    def _publish_layers(self) -> None:
        assert self.layer_stage is not None
        assert self.layer_path is not None
        assert self._layer_stage_seal is not None
        self.layer_path.mkdir()
        root_stat = self.layer_path.lstat()
        self._published_layer_root = PublishedFileIdentity(
            path=self.layer_path,
            device=root_stat.st_dev,
            inode=root_stat.st_ino,
        )
        for relative, _seal in self._layer_stage_seal.files:
            self._publish_file(
                self.layer_stage / relative,
                self.layer_path / relative,
            )
        self.layer_stage.rmdir()
        self._layer_stage_seal = None

    def _rollback_publications(self) -> None:
        if self._published_files is not None:
            for identity in reversed(self._published_files):
                remove_if_published(identity)
            self._published_files.clear()
        if self._published_layer_root is not None:
            _remove_empty_directory_if_owned(self._published_layer_root)
            self._published_layer_root = None

    @staticmethod
    def _remove_stage(seal: _StageSeal | None) -> None:
        if seal is not None:
            remove_if_published(seal.identity)

    def _remove_layer_stage(self) -> None:
        seal = self._layer_stage_seal
        if seal is None:
            return
        for _relative, file_seal in reversed(seal.files):
            remove_if_published(file_seal.identity)
        _remove_empty_directory_if_owned(seal.identity)
        self._layer_stage_seal = None


def _write_stage(path: Path, encoded: bytes) -> _StageSeal:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity: PublishedFileIdentity | None = None
    try:
        with path.open("xb") as handle:
            current = os.fstat(handle.fileno())
            identity = PublishedFileIdentity(
                path=path,
                device=current.st_dev,
                inode=current.st_ino,
            )
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return _seal_stage(path)
    except BaseException:
        if identity is not None:
            remove_if_published(identity)
        raise


def _require_directory_seal(
    seal: _DirectoryStageSeal | None,
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
    if seal.identity.path.is_symlink() or not seal.identity.path.is_dir():
        raise ProductRenderTransactionError(f"product {label} type changed")
    expected = {relative for relative, _file in seal.files}
    actual = {child.name for child in seal.identity.path.iterdir()}
    if actual != expected:
        raise ProductRenderTransactionError(f"product {label} manifest changed")
    for _relative, file_seal in seal.files:
        _require_seal(file_seal, label=f"{label} file")


def _remove_empty_directory_if_owned(identity: PublishedFileIdentity) -> bool:
    try:
        current = identity.path.lstat()
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.device, identity.inode):
        return False
    try:
        identity.path.rmdir()
    except OSError:
        return False
    return True


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


def prepare_product_render_bundle_transaction(
    input_path: Path,
    output_path: Path,
    *,
    include_recipe: bool,
    include_layers: bool,
    include_metrics: bool,
) -> ProductRenderBundleTransaction:
    """Preflight every requested product bundle entry before pixel decode."""

    source = Path(input_path)
    image = Path(output_path)
    recipe = image.with_suffix(".recipe.json") if include_recipe else None
    layers = image.parent / f"{image.stem}_layers" if include_layers else None
    metrics = image.with_suffix(".metrics.json") if include_metrics else None
    finals = [
        ("output", image),
        *([("recipe", recipe)] if recipe is not None else []),
        *([("layer root", layers)] if layers is not None else []),
        *([("metrics", metrics)] if metrics is not None else []),
    ]
    normalized = [
        _normalized_path(source),
        *[_normalized_path(path) for _, path in finals],
    ]
    if len(set(normalized)) != len(normalized):
        raise ProductRenderTransactionError(
            "product input and requested output paths must be distinct"
        )
    for label, path in finals:
        _reject_existing(path, label=label)
    token = uuid.uuid4().hex
    image_stage = image.with_name(f".{image.stem}.{token}{image.suffix}")
    recipe_stage = (
        recipe.with_name(f".{recipe.name}.{token}.stage")
        if recipe is not None
        else None
    )
    layer_stage = (
        layers.with_name(f".{layers.name}.{token}.stage")
        if layers is not None
        else None
    )
    metrics_stage = (
        metrics.with_name(f".{metrics.name}.{token}.stage")
        if metrics is not None
        else None
    )
    return ProductRenderBundleTransaction(
        image_path=image,
        image_stage=image_stage,
        recipe_path=recipe,
        recipe_stage=recipe_stage,
        layer_path=layers,
        layer_stage=layer_stage,
        metrics_path=metrics,
        metrics_stage=metrics_stage,
    )


__all__ = [
    "ProductImageRecipeTransaction",
    "ProductRenderBundleTransaction",
    "ProductRenderTransactionError",
    "preflight_product_primary_output",
    "prepare_product_image_recipe_transaction",
    "prepare_product_render_bundle_transaction",
]
