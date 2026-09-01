"""Native new-input workflow for the deterministic Look Approximation product."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    remove_if_published,
)
from src.preprocess import inspect_input

from .render_contract import sha256_file, verify_render_recipe_files
from .three_stock_preview import render_three_stock_previews_to_directory


class ProductDesktopError(ValueError):
    """Raised when the bounded native product workflow cannot proceed."""


PRODUCT_LOOKS: tuple[dict[str, str], ...] = (
    {
        "style_id": "velvia_50",
        "display_name": "Velvia 50",
        "process": "E-6 slide inspired",
        "claim": "film-inspired / Look Approximation",
    },
    {
        "style_id": "portra_400",
        "display_name": "Portra 400",
        "process": "C-41 colour-negative inspired",
        "claim": "film-inspired / Look Approximation",
    },
    {
        "style_id": "ektar_100",
        "display_name": "Ektar 100",
        "process": "C-41 colour-negative inspired",
        "claim": "film-inspired / Look Approximation",
    },
)
_LOOK_IDS = tuple(row["style_id"] for row in PRODUCT_LOOKS)
_DEFAULT_SESSION_BINDINGS = (
    "configs/render_profiles/safe_rich_v1.json",
    "configs/render_profiles/safe_rich_product_v1.json",
    "configs/color_rendering_profiles.yaml",
    "configs/film_color_stats.json",
    "configs/color_guardrails.json",
    "scripts/render_film.py",
    "src/inference/three_stock_preview.py",
    "src/inference/style_safe_engine.py",
    "src/inference/product_look_catalog.py",
    "src/inference/render_contract.py",
)

CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]], subprocess.CompletedProcess[str]
]
PreviewRenderer = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class _FileSeal:
    identity: PublishedFileIdentity
    size: int
    sha256: str


@dataclass(frozen=True)
class _DirectorySeal:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class DesktopPreviewState:
    """Hash-bound preview state eligible for one or more explicit exports."""

    input_path: Path
    input_sha256: str
    look_amount: float
    workspace: Path
    output_directory: Path
    manifest: dict[str, Any]
    workspace_seal: _DirectorySeal
    output_seal: _DirectorySeal
    files: tuple[_FileSeal, ...]
    session_bindings: tuple[_FileSeal, ...]
    root: Path
    source_commit: str


@dataclass(frozen=True)
class DesktopExportReceipt:
    """Verified final PNG16 and strict-recipe identities."""

    style_id: str
    look_amount: float
    input_path: Path
    input_sha256: str
    output_path: Path
    output_sha256: str
    recipe_path: Path
    recipe_sha256: str
    recipe: dict[str, Any]


def _run_command(
    command: Sequence[str], cwd: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    commit = completed.stdout.strip().lower()
    if (
        completed.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ProductDesktopError("source commit identity is unavailable")
    return commit


def _bounded_look_amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductDesktopError("look amount must be a finite number in [0,1]")
    amount = float(value)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ProductDesktopError("look amount must be a finite number in [0,1]")
    return amount


def _is_normal_directory(path: Path, details: os.stat_result | None = None) -> bool:
    try:
        current = path.lstat() if details is None else details
    except FileNotFoundError:
        return False
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return (
        stat.S_ISDIR(current.st_mode)
        and not path.is_symlink()
        and not int(getattr(current, "st_file_attributes", 0)) & reparse
    )


def _seal_directory(path: Path) -> _DirectorySeal:
    details = path.lstat()
    if not _is_normal_directory(path, details):
        raise ProductDesktopError("preview workspace must be a normal directory")
    return _DirectorySeal(path=path, device=details.st_dev, inode=details.st_ino)


def _directory_matches(seal: _DirectorySeal) -> bool:
    try:
        details = seal.path.lstat()
    except FileNotFoundError:
        return False
    return _is_normal_directory(seal.path, details) and (
        details.st_dev,
        details.st_ino,
    ) == (seal.device, seal.inode)


def _seal_file(path: Path) -> _FileSeal:
    details = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise ProductDesktopError("preview member must be a regular file")
    return _FileSeal(
        identity=PublishedFileIdentity(
            path=path,
            device=details.st_dev,
            inode=details.st_ino,
        ),
        size=details.st_size,
        sha256=sha256_file(path),
    )


def _file_matches(seal: _FileSeal) -> bool:
    try:
        details = seal.identity.path.lstat()
    except FileNotFoundError:
        return False
    return (
        not seal.identity.path.is_symlink()
        and stat.S_ISREG(details.st_mode)
        and (details.st_dev, details.st_ino)
        == (seal.identity.device, seal.identity.inode)
        and details.st_size == seal.size
        and sha256_file(seal.identity.path) == seal.sha256
    )


def _read_bound_file(seal: _FileSeal) -> bytes:
    try:
        with seal.identity.path.open("rb") as handle:
            details = os.fstat(handle.fileno())
            payload = handle.read(seal.size + 1)
    except OSError as exc:
        raise ProductDesktopError("preview workspace ownership changed") from exc
    if (
        (details.st_dev, details.st_ino) != (seal.identity.device, seal.identity.inode)
        or len(payload) != seal.size
        or hashlib.sha256(payload).hexdigest() != seal.sha256
    ):
        raise ProductDesktopError("preview workspace ownership changed")
    return payload


def _remove_empty_directory_if_owned(seal: _DirectorySeal) -> bool:
    if not _directory_matches(seal):
        return False
    try:
        seal.path.rmdir()
    except OSError:
        return False
    return True


class ProductDesktopWorkflow:
    """Thread-safe core for one native new-input preview/export session."""

    def __init__(
        self,
        *,
        root: Path,
        scratch_root: Path,
        python_executable: Path | None = None,
        command_runner: CommandRunner = _run_command,
        preview_renderer: PreviewRenderer = render_three_stock_previews_to_directory,
        max_preview_pixels: int = 1_000_000,
        tile_size: int = 256,
        tile_workers: int = 1,
        png_compression: int = 6,
        session_binding_paths: Sequence[Path] | None = None,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        self.scratch_root = Path(scratch_root).resolve(strict=True)
        if not self.scratch_root.is_dir():
            raise ProductDesktopError("scratch_root must be an existing directory")
        self.python_executable = Path(
            sys.executable if python_executable is None else python_executable
        ).resolve(strict=True)
        self.command_runner = command_runner
        self.preview_renderer = preview_renderer
        self.max_preview_pixels = int(max_preview_pixels)
        self.tile_size = int(tile_size)
        self.tile_workers = int(tile_workers)
        self.png_compression = int(png_compression)
        self.session_binding_paths = (
            tuple(
                (self.root / relative).resolve(strict=True)
                for relative in _DEFAULT_SESSION_BINDINGS
            )
            if session_binding_paths is None
            else tuple(
                Path(path).resolve(strict=True) for path in session_binding_paths
            )
        )
        if self.max_preview_pixels < 1 or self.tile_size < 1 or self.tile_workers < 1:
            raise ProductDesktopError("preview resource limits must be positive")
        if not 0 <= self.png_compression <= 9:
            raise ProductDesktopError("png compression must be in [0,9]")
        self._state: DesktopPreviewState | None = None
        self._lock = threading.Lock()

    @property
    def preview_state(self) -> DesktopPreviewState | None:
        return self._state

    def render_previews(
        self, input_path: Path, look_amount: float
    ) -> DesktopPreviewState:
        """Render and bind three low-resolution previews for one current input."""

        with self._lock:
            source = Path(input_path).resolve(strict=True)
            if not source.is_file():
                raise ProductDesktopError("input must be an existing regular file")
            amount = _bounded_look_amount(look_amount)
            if self._state is not None and not self._cleanup_state(self._state):
                raise ProductDesktopError(
                    "previous preview workspace ownership changed"
                )
            self._state = None
            session_bindings = tuple(
                _seal_file(path) for path in self.session_binding_paths
            )
            source_commit = _source_commit(self.root)
            workspace = self.scratch_root / (
                f"u7-10a-preview-{os.getpid()}-{uuid.uuid4().hex}"
            )
            workspace.mkdir(exist_ok=False)
            workspace_seal = _seal_directory(workspace)
            output_directory = workspace / "previews"
            state: DesktopPreviewState | None = None
            try:
                inspection = inspect_input(source)
                manifest = self.preview_renderer(
                    source,
                    output_directory,
                    root=self.root,
                    profile_path=self.root
                    / "configs/render_profiles/safe_rich_v1.json",
                    statistics_path=self.root / "configs/film_color_stats.json",
                    guardrails_path=self.root / "configs/color_guardrails.json",
                    max_preview_pixels=self.max_preview_pixels,
                    look_amount=amount,
                    seed=7,
                    tile_size=self.tile_size,
                    tile_workers=self.tile_workers,
                    png_compression=self.png_compression,
                    jpeg_scaled_decode=(
                        inspection.source_kind == "raster"
                        and source.suffix.casefold() in {".jpg", ".jpeg"}
                    ),
                    raw_half_size_decode=inspection.source_kind == "raw",
                )
                state = self._bind_preview_state(
                    source,
                    amount,
                    workspace,
                    workspace_seal,
                    output_directory,
                    manifest,
                    session_bindings,
                    source_commit,
                )
                self._validate_session(state)
            except BaseException:
                # The renderer owns and cleans its unpublished stage.  Once a
                # directory contains entries that were never bound here, this
                # layer must preserve them rather than claiming them after an
                # error merely because they appeared below our workspace.
                if state is not None:
                    self._cleanup_state(state)
                elif not output_directory.exists():
                    _remove_empty_directory_if_owned(workspace_seal)
                raise
            self._state = state
            return state

    def preview_bytes(self) -> dict[str, bytes]:
        """Return still-bound preview bytes without exposing mutable paths to UI."""

        with self._lock:
            state = self._state
            if state is None:
                raise ProductDesktopError("render previews before reading them")
            self._validate_state(state)
            self._validate_session(state)
            seals = {seal.identity.path: seal for seal in state.files}
            return {
                str(row["style_id"]): _read_bound_file(
                    seals[Path(str(row["output_path"]))]
                )
                for row in state.manifest["rows"]
            }

    def _bind_preview_state(
        self,
        source: Path,
        amount: float,
        workspace: Path,
        workspace_seal: _DirectorySeal,
        output_directory: Path,
        manifest: dict[str, Any],
        session_bindings: tuple[_FileSeal, ...],
        source_commit: str,
    ) -> DesktopPreviewState:
        if manifest.get("schema_version") != "neuro-film.three-stock-direct-preview.v1":
            raise ProductDesktopError("preview manifest schema drift")
        if manifest.get("input_sha256") != sha256_file(source):
            raise ProductDesktopError("preview input identity drift")
        if float(manifest.get("look_amount", -1.0)) != amount:
            raise ProductDesktopError("preview look amount drift")
        if int(manifest.get("preview_pixels", 0)) > self.max_preview_pixels:
            raise ProductDesktopError("preview pixel limit exceeded")
        rows = manifest.get("rows")
        if (
            not isinstance(rows, list)
            or tuple(
                str(row.get("style_id")) for row in rows if isinstance(row, Mapping)
            )
            != _LOOK_IDS
        ):
            raise ProductDesktopError("preview look catalog drift")
        output_seal = _seal_directory(output_directory)
        expected_names = {"preview.json"}
        for row in rows:
            path = Path(str(row["output_path"]))
            if path.parent != output_directory or path.name in expected_names:
                raise ProductDesktopError("preview output path escaped its workspace")
            if sha256_file(path) != row["output_sha256"]:
                raise ProductDesktopError("preview output hash drift")
            expected_names.add(path.name)
        if {entry.name for entry in output_directory.iterdir()} != expected_names:
            raise ProductDesktopError("preview member set drift")
        files = tuple(
            _seal_file(output_directory / name) for name in sorted(expected_names)
        )
        return DesktopPreviewState(
            input_path=source,
            input_sha256=sha256_file(source),
            look_amount=amount,
            workspace=workspace,
            output_directory=output_directory,
            manifest=manifest,
            workspace_seal=workspace_seal,
            output_seal=output_seal,
            files=files,
            session_bindings=session_bindings,
            root=self.root,
            source_commit=source_commit,
        )

    def export(self, style_id: str, output_path: Path) -> DesktopExportReceipt:
        """Export the selected current preview through the existing product CLI."""

        with self._lock:
            state = self._state
            if state is None:
                raise ProductDesktopError("render previews before exporting")
            if style_id not in _LOOK_IDS:
                raise ProductDesktopError("unknown or unavailable product look")
            self._validate_state(state)
            self._validate_session(state)
            if sha256_file(state.input_path) != state.input_sha256:
                raise ProductDesktopError("input changed after preview")
            destination = Path(output_path).resolve(strict=False)
            if destination.suffix.casefold() != ".png":
                raise ProductDesktopError("desktop final export requires a .png path")
            if not destination.parent.is_dir():
                raise ProductDesktopError("output parent must be an existing directory")
            recipe_path = destination.with_suffix(".recipe.json")
            if (
                destination == state.workspace
                or state.workspace in destination.parents
                or recipe_path == state.workspace
                or state.workspace in recipe_path.parents
            ):
                raise ProductDesktopError(
                    "final export must be outside the preview workspace"
                )
            if os.path.lexists(destination) or os.path.lexists(recipe_path):
                raise ProductDesktopError(
                    "output and recipe destinations must be absent"
                )
            command = self.export_command(style_id, destination)
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("PYTHON")
            }
            completed = self.command_runner(command, self.root, environment)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "render failed")[
                    -4000:
                ]
                raise ProductDesktopError(f"product export failed: {detail}")
            if not destination.is_file() or not recipe_path.is_file():
                raise ProductDesktopError(
                    "product export returned without a complete pair"
                )
            recipe = json.loads(recipe_path.read_text("utf-8"))
            verify_render_recipe_files(
                recipe,
                profile_path=self.root
                / "configs/render_profiles/safe_rich_product_v1.json",
                root=self.root,
            )
            if (
                recipe["render"]["style"] != style_id
                or float(recipe["render"]["look_amount"]) != state.look_amount
                or recipe["output"]["bit_depth"] != 16
                or str(recipe["software"]["commit"]).lower() != state.source_commit
                or recipe["claim"]["evidence_grade"] != "look-approximation"
                or recipe["claim"].get("calibrated_reference_allowed") is not False
            ):
                raise ProductDesktopError("strict product recipe semantics drift")
            return DesktopExportReceipt(
                style_id=style_id,
                look_amount=state.look_amount,
                input_path=state.input_path,
                input_sha256=state.input_sha256,
                output_path=destination,
                output_sha256=sha256_file(destination),
                recipe_path=recipe_path,
                recipe_sha256=sha256_file(recipe_path),
                recipe=recipe,
            )

    def export_command(self, style_id: str, output_path: Path) -> tuple[str, ...]:
        state = self._state
        if state is None:
            raise ProductDesktopError("render previews before exporting")
        if style_id not in _LOOK_IDS:
            raise ProductDesktopError("unknown or unavailable product look")
        return (
            str(self.python_executable),
            "-I",
            str(self.root / "scripts/render_film.py"),
            str(state.input_path),
            "--product-look",
            style_id,
            "--look-amount",
            format(state.look_amount, ".17g"),
            "--output-bit-depth",
            "16",
            "--png-compression",
            str(self.png_compression),
            "--write-recipe",
            "--tile-size",
            str(self.tile_size),
            "--tile-workers",
            str(self.tile_workers),
            "--output",
            str(Path(output_path).resolve(strict=False)),
        )

    def close(self) -> bool:
        """Remove only a still-owned, unchanged preview workspace."""

        with self._lock:
            if self._state is None:
                return True
            state = self._state
            cleaned = self._cleanup_state(state)
            if cleaned:
                self._state = None
            return cleaned

    @staticmethod
    def _validate_session(state: DesktopPreviewState) -> None:
        if _source_commit(state.root) != state.source_commit or not all(
            _file_matches(seal) for seal in state.session_bindings
        ):
            raise ProductDesktopError("preview renderer session changed")

    @staticmethod
    def _validate_state(state: DesktopPreviewState) -> None:
        try:
            member_names = {entry.name for entry in state.output_directory.iterdir()}
        except OSError as exc:
            raise ProductDesktopError("preview workspace ownership changed") from exc
        if (
            not _directory_matches(state.workspace_seal)
            or not _directory_matches(state.output_seal)
            or member_names != {seal.identity.path.name for seal in state.files}
            or not all(_file_matches(seal) for seal in state.files)
        ):
            raise ProductDesktopError("preview workspace ownership changed")

    @staticmethod
    def _cleanup_state(state: DesktopPreviewState) -> bool:
        try:
            ProductDesktopWorkflow._validate_state(state)
        except ProductDesktopError:
            return False
        removed = all(remove_if_published(seal.identity) for seal in state.files)
        return (
            removed
            and _remove_empty_directory_if_owned(state.output_seal)
            and _remove_empty_directory_if_owned(state.workspace_seal)
        )


__all__ = [
    "PRODUCT_LOOKS",
    "DesktopExportReceipt",
    "DesktopPreviewState",
    "ProductDesktopError",
    "ProductDesktopWorkflow",
]
