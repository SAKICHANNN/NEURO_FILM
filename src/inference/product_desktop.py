"""Native new-input workflow for the deterministic Look Approximation product."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    remove_if_published,
)
from src.preprocess import inspect_input

from .render_contract import (
    sha256_file,
    validate_render_recipe,
    verify_render_recipe_files,
    verify_render_recipe_inputs,
)
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
PRODUCT_PREVIEW_DISPLAY_SIZE = (300, 260)
_LOOK_IDS = tuple(row["style_id"] for row in PRODUCT_LOOKS)
_DEFAULT_SESSION_BINDINGS = (
    "configs/render_profiles/safe_rich_v1.json",
    "configs/render_profiles/safe_rich_product_v1.json",
    "configs/color_rendering_profiles.yaml",
    "configs/film_color_stats.json",
    "configs/color_guardrails.json",
    "scripts/render_film.py",
    "src/inference/product_detail_inspection.py",
    "src/inference/three_stock_preview.py",
    "src/inference/style_safe_engine.py",
    "src/inference/product_look_catalog.py",
    "src/inference/render_contract.py",
)

CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]], subprocess.CompletedProcess[str]
]
PreviewRenderer = Callable[..., dict[str, Any]]
BatchProgress = Callable[[int, int, str], None]


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
class ProductOutputFormat:
    """Existing product encoder exposed by the one-photo desktop."""

    format_id: str
    display_name: str
    canonical_extension: str
    accepted_extensions: tuple[str, ...]
    recipe_format: str
    bit_depth: int
    png_compression: int | None


PRODUCT_OUTPUT_FORMATS: tuple[ProductOutputFormat, ...] = (
    ProductOutputFormat(
        format_id="png16",
        display_name="PNG16",
        canonical_extension=".png",
        accepted_extensions=(".png",),
        recipe_format="PNG",
        bit_depth=16,
        png_compression=6,
    ),
    ProductOutputFormat(
        format_id="tiff16",
        display_name="TIFF16",
        canonical_extension=".tiff",
        accepted_extensions=(".tif", ".tiff"),
        recipe_format="TIFF",
        bit_depth=16,
        png_compression=None,
    ),
    ProductOutputFormat(
        format_id="jpeg8",
        display_name="JPEG8",
        canonical_extension=".jpg",
        accepted_extensions=(".jpg", ".jpeg"),
        recipe_format="JPEG",
        bit_depth=8,
        png_compression=None,
    ),
)
_OUTPUT_FORMATS_BY_ID = {row.format_id: row for row in PRODUCT_OUTPUT_FORMATS}
_DEFAULT_OUTPUT_FORMAT_ID = "png16"


def product_output_format(output_format_id: str) -> ProductOutputFormat:
    """Resolve one frozen desktop output format or fail closed."""

    if not isinstance(output_format_id, str):
        raise ProductDesktopError("unknown desktop output format")
    try:
        return _OUTPUT_FORMATS_BY_ID[output_format_id]
    except KeyError as exc:
        raise ProductDesktopError("unknown desktop output format") from exc


def _validate_output_format_path(
    output_format_id: str, output_path: Path
) -> ProductOutputFormat:
    spec = product_output_format(output_format_id)
    suffix = Path(output_path).suffix.casefold()
    if not suffix:
        raise ProductDesktopError("desktop final export requires a file extension")
    if suffix not in spec.accepted_extensions:
        raise ProductDesktopError(
            f"desktop {spec.display_name} export requires "
            f"{', '.join(spec.accepted_extensions)}"
        )
    return spec


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
    """Verified final image and strict-recipe identities."""

    style_id: str
    look_amount: float
    input_path: Path
    input_sha256: str
    output_path: Path
    output_sha256: str
    recipe_path: Path
    recipe_sha256: str
    recipe: dict[str, Any]
    output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID


@dataclass(frozen=True)
class DesktopBatchInput:
    """Canonical source identity frozen before representative preview."""

    path: Path
    basename: str
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen=True)
class DesktopBatchReceipt:
    """Verified aggregate identity for one single-look desktop batch."""

    batch_id: str
    style_id: str
    look_amount: float
    output_directory: Path
    receipt_path: Path
    receipt_sha256: str
    job_count: int
    receipt: dict[str, Any]
    output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID


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


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path.resolve(strict=False))))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_output_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem).strip("-_")
    return (stem[:48] or "image").casefold()


def _valid_windows_component(name: str) -> bool:
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    base = name.split(".", 1)[0].casefold()
    return bool(
        name
        and name not in {".", ".."}
        and not name.endswith((" ", "."))
        and base not in reserved
        and not any(character in '<>:"/\\|?*' for character in name)
    )


def _batch_directory_rename_supported() -> bool:
    """Return whether directory rename is create-only on the formal platform."""

    return os.name == "nt"


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


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


def _batch_input_matches(source: DesktopBatchInput) -> bool:
    try:
        details = source.path.lstat()
    except FileNotFoundError:
        return False
    return (
        not source.path.is_symlink()
        and stat.S_ISREG(details.st_mode)
        and (details.st_dev, details.st_ino) == (source.device, source.inode)
        and details.st_size == source.size
        and sha256_file(source.path) == source.sha256
    )


def _write_bound_json(path: Path, payload: Mapping[str, Any]) -> _FileSeal:
    encoded = _encode_json(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o666,
    )
    details = os.fstat(descriptor)
    identity = PublishedFileIdentity(
        path=path,
        device=details.st_dev,
        inode=details.st_ino,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            if handle.write(encoded) != len(encoded):
                raise OSError("short batch JSON write")
            handle.flush()
            os.fsync(handle.fileno())
        seal = _seal_file(path)
        if (
            seal.identity != identity
            or seal.size != len(encoded)
            or seal.sha256 != hashlib.sha256(encoded).hexdigest()
        ):
            raise ProductDesktopError("batch JSON publication identity drifted")
        return seal
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        remove_if_published(identity)
        raise


def _replace_owned_json(path: Path, payload: Mapping[str, Any]) -> _FileSeal:
    details = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ProductDesktopError("batch recipe ownership changed")
    encoded = _encode_json(payload)
    identity = PublishedFileIdentity(
        path=path,
        device=details.st_dev,
        inode=details.st_ino,
    )
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "r+b") as handle:
            descriptor = -1
            current = os.fstat(handle.fileno())
            if (
                (current.st_dev, current.st_ino)
                != (identity.device, identity.inode)
                or current.st_nlink != 1
            ):
                raise ProductDesktopError("batch recipe ownership changed")
            handle.seek(0)
            if handle.write(encoded) != len(encoded):
                raise OSError("short batch recipe write")
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        seal = _seal_file(path)
        if (
            seal.identity != identity
            or seal.size != len(encoded)
            or seal.sha256 != hashlib.sha256(encoded).hexdigest()
        ):
            raise ProductDesktopError("batch recipe ownership changed")
        return seal
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        remove_if_published(identity)
        raise


def _cleanup_bound_stage(
    stage: _DirectorySeal,
    files: Sequence[_FileSeal],
) -> bool:
    for seal in reversed(files):
        remove_if_published(seal.identity)
    return _remove_empty_directory_if_owned(stage)


def _bind_unfinished_child_candidates(
    stage: _DirectorySeal,
    paths: Sequence[Path],
) -> tuple[_FileSeal, ...]:
    """Bind regular expected outputs created inside our unguessable owned stage."""

    if not _directory_matches(stage):
        return ()
    bound: list[_FileSeal] = []
    for path in paths:
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        if (
            path.parent == stage.path
            and not path.is_symlink()
            and stat.S_ISREG(details.st_mode)
            and details.st_nlink == 1
        ):
            bound.append(_seal_file(path))
    return tuple(bound)


def _bind_file_after_directory_rename(seal: _FileSeal, root: Path) -> _FileSeal:
    """Bind an exact renamed member on filesystems that synthesize new inodes."""

    rebound = _seal_file(root / seal.identity.path.name)
    if rebound.size != seal.size or rebound.sha256 != seal.sha256:
        raise ProductDesktopError("published batch member identity drifted")
    return rebound


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

    def bind_batch_inputs(
        self, input_paths: Sequence[Path]
    ) -> tuple[DesktopBatchInput, ...]:
        """Hash and canonically bind one bounded set before representative preview."""

        if isinstance(input_paths, (str, bytes)):
            raise ProductDesktopError("batch inputs must be a path sequence")
        if not 1 <= len(input_paths) <= 100:
            raise ProductDesktopError("choose between one and 100 photos")
        resolved = [Path(path).resolve(strict=True) for path in input_paths]
        ordered = sorted(resolved, key=_normalized_path)
        normalized = [_normalized_path(path) for path in ordered]
        if len(set(normalized)) != len(normalized):
            raise ProductDesktopError("batch inputs must identify unique paths")
        identities: set[tuple[int, int]] = set()
        bound: list[DesktopBatchInput] = []
        for source in ordered:
            details = source.lstat()
            if source.is_symlink() or not stat.S_ISREG(details.st_mode):
                raise ProductDesktopError("batch input must be a regular file")
            identity = (details.st_dev, details.st_ino)
            if identity in identities:
                raise ProductDesktopError("batch inputs must identify unique files")
            identities.add(identity)
            inspect_input(source)
            bound.append(
                DesktopBatchInput(
                    path=source,
                    basename=source.name,
                    sha256=sha256_file(source),
                    size=details.st_size,
                    device=details.st_dev,
                    inode=details.st_ino,
                )
            )
        return tuple(bound)

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
                    max_preview_width=PRODUCT_PREVIEW_DISPLAY_SIZE[0],
                    max_preview_height=PRODUCT_PREVIEW_DISPLAY_SIZE[1],
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
                    include_input_preview=True,
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

    def input_preview_bytes(self) -> bytes:
        """Return the bound generic input-basis preview for the native UI."""

        with self._lock:
            state = self._state
            if state is None:
                raise ProductDesktopError("render previews before reading them")
            self._validate_state(state)
            self._validate_session(state)
            input_preview = state.manifest.get("input_preview")
            if not isinstance(input_preview, Mapping):
                raise ProductDesktopError("input preview is unavailable")
            path = Path(str(input_preview.get("output_path", "")))
            seals = {seal.identity.path: seal for seal in state.files}
            seal = seals.get(path)
            if seal is None:
                raise ProductDesktopError("input preview binding is unavailable")
            return _read_bound_file(seal)

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
        preview_width = manifest.get("preview_width")
        preview_height = manifest.get("preview_height")
        preview_pixels = manifest.get("preview_pixels")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (preview_width, preview_height, preview_pixels)
        ):
            raise ProductDesktopError("preview geometry drift")
        if preview_pixels != preview_width * preview_height:
            raise ProductDesktopError("preview geometry drift")
        if preview_pixels > self.max_preview_pixels:
            raise ProductDesktopError("preview pixel limit exceeded")
        if (
            preview_width > PRODUCT_PREVIEW_DISPLAY_SIZE[0]
            or preview_height > PRODUCT_PREVIEW_DISPLAY_SIZE[1]
        ):
            raise ProductDesktopError("preview display bounds drift")
        if (
            manifest.get("max_preview_width") != PRODUCT_PREVIEW_DISPLAY_SIZE[0]
            or manifest.get("max_preview_height")
            != PRODUCT_PREVIEW_DISPLAY_SIZE[1]
        ):
            raise ProductDesktopError("preview display request drift")
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
        input_preview = manifest.get("input_preview")
        if not isinstance(input_preview, Mapping) or set(input_preview) != {
            "role",
            "output_path",
            "output_sha256",
            "source_kind",
            "display_adapter",
            "claim",
        }:
            raise ProductDesktopError("input preview contract drift")
        input_path = Path(str(input_preview["output_path"]))
        if (
            input_preview["role"] != "input_basis"
            or input_path.parent != output_directory
            or input_path.name != "input.preview.png"
            or input_preview["source_kind"] not in {"raster", "raw"}
            or input_preview["display_adapter"]
            != "existing WorkingImage to display-sRGB adapter"
            or input_preview["claim"]
            != "generic display adapter, not a calibrated camera rendering"
            or sha256_file(input_path) != input_preview["output_sha256"]
        ):
            raise ProductDesktopError("input preview contract drift")
        expected_names.add(input_path.name)
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

    def export(
        self,
        style_id: str,
        output_path: Path,
        *,
        output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID,
    ) -> DesktopExportReceipt:
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
            format_spec = _validate_output_format_path(output_format_id, destination)
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
            command = self.export_command(
                style_id,
                destination,
                output_format_id=output_format_id,
            )
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
                or recipe["output"]["format"] != format_spec.recipe_format
                or recipe["output"]["bit_depth"] != format_spec.bit_depth
                or Path(str(recipe["output"]["path"])).resolve() != destination
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
                output_format_id=output_format_id,
            )

    def render_batch_previews(
        self,
        input_paths: Sequence[Path],
        look_amount: float,
        *,
        representative_path: Path | None = None,
    ) -> tuple[DesktopPreviewState, tuple[DesktopBatchInput, ...]]:
        """Bind every source, then preview one explicit member or the canonical first."""

        bound = self.bind_batch_inputs(input_paths)
        representative = bound[0]
        if representative_path is not None:
            try:
                candidate = Path(representative_path)
                candidate_details = candidate.lstat()
                reparse = int(
                    getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                )
                if candidate.is_symlink() or int(
                    getattr(candidate_details, "st_file_attributes", 0)
                ) & reparse:
                    raise ProductDesktopError(
                        "batch representative must be one selected photo"
                    )
                resolved = candidate.resolve(strict=True)
                raw_absolute = os.path.normcase(
                    os.path.abspath(os.fspath(candidate))
                )
                if raw_absolute != _normalized_path(resolved):
                    raise ProductDesktopError(
                        "batch representative must be one selected photo"
                    )
                details = resolved.lstat()
            except OSError as exc:
                raise ProductDesktopError(
                    "batch representative must be one selected photo"
                ) from exc
            matches = tuple(
                row
                for row in bound
                if row.path == resolved
                and (row.device, row.inode) == (details.st_dev, details.st_ino)
            )
            if len(matches) != 1:
                raise ProductDesktopError(
                    "batch representative must be one selected photo"
                )
            representative = matches[0]
        if not _batch_input_matches(representative):
            raise ProductDesktopError("batch representative changed before preview")
        state = self.render_previews(representative.path, look_amount)
        if (
            state.input_path != representative.path
            or state.input_sha256 != representative.sha256
        ):
            self.close()
            raise ProductDesktopError("representative preview input identity drifted")
        return state, bound

    def export_batch(
        self,
        inputs: Sequence[DesktopBatchInput],
        style_id: str,
        output_directory: Path,
        *,
        output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID,
        cancel_event: threading.Event | None = None,
        progress: BatchProgress | None = None,
    ) -> DesktopBatchReceipt:
        """Publish one atomic single-look directory through the existing CLI."""

        format_spec = product_output_format(output_format_id)
        with self._lock:
            state = self._state
            if state is None:
                raise ProductDesktopError("render previews before exporting")
            if style_id not in _LOOK_IDS:
                raise ProductDesktopError("unknown or unavailable product look")
            rows = tuple(inputs)
            if not 2 <= len(rows) <= 100:
                raise ProductDesktopError("desktop batch requires two to 100 photos")
            if tuple(sorted((row.path for row in rows), key=_normalized_path)) != tuple(
                row.path for row in rows
            ):
                raise ProductDesktopError("batch input order is not canonical")
            if len({_normalized_path(row.path) for row in rows}) != len(rows) or len(
                {(row.device, row.inode) for row in rows}
            ) != len(rows):
                raise ProductDesktopError("batch inputs must identify unique files")
            representative_matches = tuple(
                row
                for row in rows
                if state.input_path == row.path and state.input_sha256 == row.sha256
            )
            if len(representative_matches) != 1:
                raise ProductDesktopError(
                    "representative preview does not bind this batch"
                )
            self._validate_state(state)
            self._validate_session(state)
            if not all(_batch_input_matches(row) for row in rows):
                raise ProductDesktopError("batch input changed after preview")
            if not _batch_directory_rename_supported():
                raise ProductDesktopError("desktop batch publication requires Windows")

            destination = Path(output_directory).resolve(strict=False)
            if not _valid_windows_component(destination.name):
                raise ProductDesktopError("batch destination name is unsafe on Windows")
            if not destination.parent.is_dir():
                raise ProductDesktopError("batch destination parent must exist")
            if os.path.lexists(destination):
                raise ProductDesktopError("batch destination must be absent")
            if destination == state.workspace or state.workspace in destination.parents:
                raise ProductDesktopError(
                    "batch destination must be outside the preview workspace"
                )
            stage = destination.with_name(
                f".{destination.name}.u7-11a-{uuid.uuid4().hex}.stage"
            )
            if os.path.lexists(stage):
                raise ProductDesktopError("batch stage path unexpectedly exists")
            stage.mkdir()
            stage_seal = _seal_directory(stage)
            owned_files: list[_FileSeal] = []
            published = False
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("PYTHON")
            }
            receipt_jobs: list[dict[str, Any]] = []
            expected_names: set[str] = set()
            try:
                for index, source in enumerate(rows, 1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise ProductDesktopError("batch cancelled safely")
                    self._validate_state(state)
                    self._validate_session(state)
                    if not _batch_input_matches(source):
                        raise ProductDesktopError(
                            f"batch input changed before child {index}"
                        )
                    stem = f"{index:04d}-{_safe_output_stem(source.path)}-{style_id}"
                    image_name = f"{stem}{format_spec.canonical_extension}"
                    recipe_name = f"{stem}.recipe.json"
                    image_path = stage / image_name
                    recipe_path = stage / recipe_name
                    expected_names.update({image_name, recipe_name})
                    try:
                        completed = self.command_runner(
                            self._build_export_command(
                                source.path,
                                style_id,
                                image_path,
                                state.look_amount,
                                output_format_id=output_format_id,
                            ),
                            self.root,
                            environment,
                        )
                    except BaseException:
                        owned_files.extend(
                            _bind_unfinished_child_candidates(
                                stage_seal, (image_path, recipe_path)
                            )
                        )
                        raise
                    if completed.returncode != 0:
                        owned_files.extend(
                            _bind_unfinished_child_candidates(
                                stage_seal, (image_path, recipe_path)
                            )
                        )
                        detail = (
                            completed.stderr or completed.stdout or "render failed"
                        )[-4000:]
                        raise ProductDesktopError(
                            f"batch child {index} failed: {detail}"
                        )
                    if not image_path.is_file() or not recipe_path.is_file():
                        owned_files.extend(
                            _bind_unfinished_child_candidates(
                                stage_seal, (image_path, recipe_path)
                            )
                        )
                        raise ProductDesktopError(
                            f"batch child {index} returned without a complete pair"
                        )
                    image_seal = _seal_file(image_path)
                    recipe_seal = _seal_file(recipe_path)
                    owned_files.extend((image_seal, recipe_seal))
                    recipe = json.loads(recipe_path.read_text("utf-8"))
                    verify_render_recipe_files(
                        recipe,
                        profile_path=self.root
                        / "configs/render_profiles/safe_rich_product_v1.json",
                        root=self.root,
                    )
                    final_image = destination / image_name
                    if (
                        recipe["input"]["path"] != str(source.path)
                        or recipe["input"]["sha256"] != source.sha256
                        or recipe["render"]["style"] != style_id
                        or float(recipe["render"]["look_amount"])
                        != state.look_amount
                        or recipe["output"]["format"] != format_spec.recipe_format
                        or recipe["output"]["bit_depth"] != format_spec.bit_depth
                        or recipe["output"]["sha256"] != image_seal.sha256
                        or str(recipe["software"]["commit"]).lower()
                        != state.source_commit
                        or recipe["claim"]["evidence_grade"]
                        != "look-approximation"
                        or recipe["claim"].get("calibrated_reference_allowed")
                        is not False
                    ):
                        raise ProductDesktopError(
                            f"batch child {index} recipe semantics drifted"
                        )
                    recipe["output"]["path"] = str(
                        final_image.resolve(strict=False)
                    )
                    validate_render_recipe(recipe)
                    verify_render_recipe_inputs(
                        recipe,
                        profile_path=self.root
                        / "configs/render_profiles/safe_rich_product_v1.json",
                        root=self.root,
                    )
                    owned_files[-1] = _replace_owned_json(recipe_path, recipe)
                    receipt_jobs.append(
                        {
                            "job_id": f"{index:04d}",
                            "input_basename": source.basename,
                            "input_sha256": source.sha256,
                            "output_path": PurePosixPath(image_name).as_posix(),
                            "output_sha256": image_seal.sha256,
                            "recipe_path": PurePosixPath(recipe_name).as_posix(),
                            "recipe_sha256": owned_files[-1].sha256,
                        }
                    )
                    if progress is not None:
                        progress(index, len(rows), source.basename)

                if cancel_event is not None and cancel_event.is_set():
                    raise ProductDesktopError("batch cancelled safely")
                self._validate_state(state)
                self._validate_session(state)
                if _source_commit(self.root) != state.source_commit:
                    raise ProductDesktopError("source commit changed during batch")
                if not all(_batch_input_matches(row) for row in rows):
                    raise ProductDesktopError("batch input changed before publication")
                if {entry.name for entry in stage.iterdir()} != expected_names:
                    raise ProductDesktopError("batch stage member set drifted")
                identity = {
                    "schema_version": (
                        "kmcfm.desktop-single-look-batch.v1"
                        if output_format_id == _DEFAULT_OUTPUT_FORMAT_ID
                        else "kmcfm.desktop-single-look-batch.v2"
                    ),
                    "style_id": style_id,
                    "look_amount": state.look_amount,
                    "source_commit": state.source_commit,
                    "job_count": len(rows),
                    "output_format": format_spec.recipe_format,
                    "output_bit_depth": format_spec.bit_depth,
                    "jobs": receipt_jobs,
                    "claim": {
                        "output_label": "film-inspired / Look Approximation",
                        "evidence_grade": "look-approximation",
                        "calibrated_stock_response": False,
                        "physical_film_reproduction": False,
                        "stock_distinguishability": False,
                    },
                }
                if output_format_id != _DEFAULT_OUTPUT_FORMAT_ID:
                    identity["output_format_id"] = output_format_id
                receipt = {"batch_id": _canonical_sha256(identity), **identity}
                receipt_path = stage / "batch.json"
                owned_files.append(_write_bound_json(receipt_path, receipt))
                expected_names.add("batch.json")
                if {entry.name for entry in stage.iterdir()} != expected_names:
                    raise ProductDesktopError("batch receipt member set drifted")
                if os.path.lexists(destination):
                    raise ProductDesktopError("batch destination appeared")
                os.rename(stage, destination)
                final_root: _DirectorySeal | None = None
                final_files: list[_FileSeal] = []
                try:
                    final_root = _seal_directory(destination)
                    if final_root.device != stage_seal.device:
                        raise ProductDesktopError(
                            "published batch directory identity drifted"
                        )
                    binding_failed = False
                    for seal in owned_files:
                        try:
                            final_files.append(
                                _bind_file_after_directory_rename(seal, destination)
                            )
                        except (OSError, ProductDesktopError):
                            binding_failed = True
                    if binding_failed or not all(
                        _file_matches(seal) for seal in final_files
                    ):
                        raise ProductDesktopError(
                            "published batch member identity drifted"
                        )
                    if {entry.name for entry in destination.iterdir()} != expected_names:
                        raise ProductDesktopError(
                            "published batch member set drifted"
                        )
                    for row in receipt_jobs:
                        recipe_path = destination / row["recipe_path"]
                        recipe = json.loads(recipe_path.read_text("utf-8"))
                        verify_render_recipe_files(
                            recipe,
                            profile_path=self.root
                            / "configs/render_profiles/safe_rich_product_v1.json",
                            root=self.root,
                        )
                        if (
                            sha256_file(recipe_path) != row["recipe_sha256"]
                            or sha256_file(destination / row["output_path"])
                            != row["output_sha256"]
                        ):
                            raise ProductDesktopError(
                                "published batch child identity drifted"
                            )
                    final_receipt = destination / "batch.json"
                    if (
                        json.loads(final_receipt.read_text("utf-8")) != receipt
                        or sha256_file(final_receipt) != final_files[-1].sha256
                    ):
                        raise ProductDesktopError(
                            "published aggregate receipt identity drifted"
                        )
                except BaseException:
                    if final_root is not None:
                        _cleanup_bound_stage(final_root, final_files)
                    raise
                published = True
                return DesktopBatchReceipt(
                    batch_id=receipt["batch_id"],
                    style_id=style_id,
                    look_amount=state.look_amount,
                    output_directory=destination,
                    receipt_path=final_receipt,
                    receipt_sha256=sha256_file(final_receipt),
                    job_count=len(rows),
                    receipt=receipt,
                    output_format_id=output_format_id,
                )
            finally:
                if not published:
                    _cleanup_bound_stage(stage_seal, owned_files)

    def export_command(
        self,
        style_id: str,
        output_path: Path,
        *,
        output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID,
    ) -> tuple[str, ...]:
        state = self._state
        if state is None:
            raise ProductDesktopError("render previews before exporting")
        if style_id not in _LOOK_IDS:
            raise ProductDesktopError("unknown or unavailable product look")
        return self._build_export_command(
            state.input_path,
            style_id,
            Path(output_path),
            state.look_amount,
            output_format_id=output_format_id,
        )

    def _build_export_command(
        self,
        input_path: Path,
        style_id: str,
        output_path: Path,
        look_amount: float,
        *,
        output_format_id: str = _DEFAULT_OUTPUT_FORMAT_ID,
    ) -> tuple[str, ...]:
        format_spec = _validate_output_format_path(output_format_id, output_path)
        command = [
            str(self.python_executable),
            "-I",
            str(self.root / "scripts/render_film.py"),
            str(Path(input_path).resolve(strict=True)),
            "--product-look",
            style_id,
            "--look-amount",
            format(_bounded_look_amount(look_amount), ".17g"),
            "--output-bit-depth",
            str(format_spec.bit_depth),
        ]
        if format_spec.png_compression is not None:
            command.extend(("--png-compression", str(self.png_compression)))
        command.extend(
            (
                "--write-recipe",
                "--tile-size",
                str(self.tile_size),
                "--tile-workers",
                str(self.tile_workers),
                "--output",
                str(Path(output_path).resolve(strict=False)),
            )
        )
        return tuple(command)

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
    "PRODUCT_PREVIEW_DISPLAY_SIZE",
    "DesktopBatchInput",
    "DesktopBatchReceipt",
    "DesktopExportReceipt",
    "DesktopPreviewState",
    "ProductDesktopError",
    "ProductDesktopWorkflow",
]
