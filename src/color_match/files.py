"""Transactional SDR file adapter for uploaded-reference image matching."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import uuid

import numpy as np

from src.inference.render_contract import atomic_write_json, sha256_file
from src.preprocess import (
    load_working_image,
    save_rec2020_16_png,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    WorkingImage,
    working_image_to_srgb_float,
)

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .contracts import (
    ReferenceLookPolicy,
    ReferenceLookRecipe,
    ReferenceMatchContractError,
)
from .fit import fit_reference_look
from .output_metadata_policy import attest_reference_file_output_metadata
from .render import ReferenceMatchDiagnostics
from .replay import (
    load_reference_look_recipe_bound,
    save_reference_look_recipe,
)
from .safety import (
    ReferenceRenderGuardPolicy,
    ReferenceSafetyDecision,
    render_reference_look_guarded,
)
from .transaction_lock import target_transaction_lock


_SDR_OUTPUT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
_SDR16_OUTPUT_EXTENSIONS = frozenset({".png", ".tif", ".tiff"})
REFERENCE_FILE_OUTPUT_CAPABILITIES_ID = (
    "neuro-film.reference-file-output-capabilities.v1"
)
REFERENCE_FILE_INPUT_INSPECTION_SCHEMA_ID = (
    "neuro-film.reference-file-input-inspection.v1"
)
REFERENCE_FILE_INPUT_INSPECTION_BATCH_SCHEMA_ID = (
    "neuro-film.reference-file-input-inspection-batch.v1"
)
REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING = (
    "preflight-only-not-render-authorization"
)
_SUPPORTED_FILE_INPUT_RAILS = frozenset(
    {
        ("linear_srgb", "display_linear"),
        ("linear_rec2020", "display_linear"),
    }
)


@dataclass(frozen=True)
class FileReferenceOutputCapability:
    """One exact output rail exposed to product clients before rendering."""

    working_space: str
    transfer_state: str
    output_bit_depth: int
    extensions: tuple[str, ...]
    encoding_profile: str


@dataclass(frozen=True)
class FileReferenceInputInspection:
    """Bounded decoded-rail preflight; never render authorization."""

    schema_id: str
    claim_ceiling: str
    path: Path
    file_sha256: str | None
    accepted: bool
    working_space: str | None
    transfer_state: str | None
    failure_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "claim_ceiling": self.claim_ceiling,
            "path": str(self.path.resolve(strict=False)),
            "file_sha256": self.file_sha256,
            "accepted": self.accepted,
            "working_space": self.working_space,
            "transfer_state": self.transfer_state,
            "failure_code": self.failure_code,
        }


def inspect_reference_file_input(
    path: Path | str,
) -> FileReferenceInputInspection:
    """Decode and hash one file for advisory product compatibility preflight."""

    source = Path(path)
    common = {
        "schema_id": REFERENCE_FILE_INPUT_INSPECTION_SCHEMA_ID,
        "claim_ceiling": REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING,
        "path": source,
    }
    if not source.is_file():
        return FileReferenceInputInspection(
            **common,
            file_sha256=None,
            accepted=False,
            working_space=None,
            transfer_state=None,
            failure_code="not-a-file",
        )
    try:
        before = sha256_file(source)
    except OSError:
        return FileReferenceInputInspection(
            **common,
            file_sha256=None,
            accepted=False,
            working_space=None,
            transfer_state=None,
            failure_code="file-changed-during-inspection",
        )
    try:
        image = load_working_image(source)
    except Exception:  # noqa: BLE001 - advisory preflight must be structured.
        return FileReferenceInputInspection(
            **common,
            file_sha256=before,
            accepted=False,
            working_space=None,
            transfer_state=None,
            failure_code="decode-or-color-state-rejected",
        )
    try:
        after = sha256_file(source)
    except OSError:
        after = None
    if after != before:
        return FileReferenceInputInspection(
            **common,
            file_sha256=None,
            accepted=False,
            working_space=None,
            transfer_state=None,
            failure_code="file-changed-during-inspection",
        )
    accepted = (
        image.working_space,
        image.transfer_state,
    ) in _SUPPORTED_FILE_INPUT_RAILS
    return FileReferenceInputInspection(
        **common,
        file_sha256=before,
        accepted=accepted,
        working_space=image.working_space,
        transfer_state=image.transfer_state,
        failure_code=None if accepted else "unsupported-decoded-rail",
    )


def inspect_reference_file_inputs(
    paths: Iterable[Path | str],
) -> dict[str, Any]:
    """Inspect a bounded ordered batch without granting render authority."""

    sources = _paths(paths, "input_paths")
    return {
        "schema_id": REFERENCE_FILE_INPUT_INSPECTION_BATCH_SCHEMA_ID,
        "claim_ceiling": REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING,
        "inspections": [
            inspect_reference_file_input(path).to_dict()
            for path in sources
        ],
    }


def reference_file_supported_input_rails() -> tuple[dict[str, str], ...]:
    """Return the decoded colour rails accepted by the render transaction."""

    return tuple(
        {
            "working_space": working_space,
            "transfer_state": transfer_state,
        }
        for working_space, transfer_state in sorted(_SUPPORTED_FILE_INPUT_RAILS)
    )


def reference_file_output_capabilities(
) -> tuple[FileReferenceOutputCapability, ...]:
    """Return the immutable v1 file-output support matrix."""

    return (
        FileReferenceOutputCapability(
            working_space="linear_srgb",
            transfer_state="display_linear",
            output_bit_depth=8,
            extensions=tuple(sorted(_SDR_OUTPUT_EXTENSIONS)),
            encoding_profile="srgb-icc.v1",
        ),
        FileReferenceOutputCapability(
            working_space="linear_srgb",
            transfer_state="display_linear",
            output_bit_depth=16,
            extensions=tuple(sorted(_SDR16_OUTPUT_EXTENSIONS)),
            encoding_profile="srgb-icc.v1",
        ),
        FileReferenceOutputCapability(
            working_space="linear_rec2020",
            transfer_state="display_linear",
            output_bit_depth=16,
            extensions=(".png",),
            encoding_profile="bt2020-sdr-cicp-1-1-0-1.v1",
        ),
    )


def reference_file_output_capabilities_payload() -> dict[str, Any]:
    """Return the strict JSON-compatible v1 capability envelope."""

    return {
        "schema_id": REFERENCE_FILE_OUTPUT_CAPABILITIES_ID,
        "capabilities": [
            {
                "working_space": row.working_space,
                "transfer_state": row.transfer_state,
                "output_bit_depth": row.output_bit_depth,
                "extensions": list(row.extensions),
                "encoding_profile": row.encoding_profile,
            }
            for row in reference_file_output_capabilities()
        ],
    }


def resolve_reference_file_output_capability(
    *,
    working_space: str,
    transfer_state: str,
    output_bit_depth: int,
    output_extension: str,
) -> FileReferenceOutputCapability:
    """Resolve one exact output choice against the public v1 matrix."""

    if not isinstance(working_space, str) or not working_space:
        raise ReferenceMatchContractError("working_space must be non-empty")
    if not isinstance(transfer_state, str) or not transfer_state:
        raise ReferenceMatchContractError("transfer_state must be non-empty")
    if (
        isinstance(output_bit_depth, bool)
        or not isinstance(output_bit_depth, int)
    ):
        raise ReferenceMatchContractError("output_bit_depth must be 8 or 16")
    if (
        not isinstance(output_extension, str)
        or not output_extension.startswith(".")
        or output_extension != output_extension.strip()
    ):
        raise ReferenceMatchContractError(
            "output_extension must be a dot-prefixed extension"
        )
    normalized_extension = output_extension.casefold()
    for capability in reference_file_output_capabilities():
        if (
            capability.working_space == working_space
            and capability.transfer_state == transfer_state
            and capability.output_bit_depth == output_bit_depth
            and normalized_extension in capability.extensions
        ):
            return capability
    if working_space == "linear_rec2020":
        raise ReferenceMatchContractError(
            "linear_rec2020 file output requires 16-bit PNG"
        )
    raise ReferenceMatchContractError(
        "requested file-output rail is unsupported"
    )


@dataclass(frozen=True)
class FileReferenceMatchOutput:
    """Durable output identity and render diagnostics for one source file."""

    source_path: Path
    source_file_sha256: str
    output_path: Path
    output_sha256: str
    output_format: str
    output_bit_depth: int
    encode_clipped_fraction: float
    diagnostics: ReferenceMatchDiagnostics
    safety: ReferenceSafetyDecision


@dataclass(frozen=True)
class FileReferenceMatchResult:
    """One fitted recipe plus all committed output identities."""

    reference_path: Path
    reference_file_sha256: str
    recipe: ReferenceLookRecipe
    recipe_path: Path | None
    recipe_file_sha256: str | None
    outputs: tuple[FileReferenceMatchOutput, ...]
    report_path: Path | None = None
    report_file_sha256: str | None = None


@dataclass(frozen=True)
class FileReferenceReplayResult:
    """One verified stored recipe plus all replayed output identities."""

    recipe: ReferenceLookRecipe
    recipe_path: Path
    recipe_file_sha256: str
    outputs: tuple[FileReferenceMatchOutput, ...]
    report_path: Path | None = None
    report_file_sha256: str | None = None


_ReportPayloadFactory = Callable[
    [tuple[FileReferenceMatchOutput, ...], str | None],
    Mapping[str, Any],
]


def _paths(values: Iterable[Path | str], label: str) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes, Path)):
        raise ReferenceMatchContractError(f"{label} must be an iterable of paths")
    paths: list[Path] = []
    try:
        for value in values:
            if len(paths) >= MAX_REFERENCE_MATCH_BATCH_SOURCES:
                raise ReferenceMatchContractError(
                    "file reference match supports at most "
                    f"{MAX_REFERENCE_MATCH_BATCH_SOURCES} sources"
                )
            paths.append(Path(value))
    except ReferenceMatchContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            f"{label} must be an iterable of paths"
        ) from exc
    if not paths:
        raise ReferenceMatchContractError(f"{label} must not be empty")
    return tuple(paths)


def _resolved_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _strict_path_ancestor(ancestor: str, descendant: str) -> bool:
    if ancestor == descendant:
        return False
    try:
        return os.path.commonpath((ancestor, descendant)) == ancestor
    except ValueError:
        # Different Windows drives cannot be nested.
        return False


def _validate_run_path_topology(
    *,
    protected_paths: tuple[Path, ...],
    destination_paths: tuple[Path, ...],
) -> None:
    """Reject file destinations that require another run file to be a directory."""

    for destination in destination_paths:
        if destination.exists() and destination.is_dir():
            raise ReferenceMatchContractError(
                f"run destination must not be a directory: {destination}"
            )
    protected = tuple(
        (_resolved_key(path), path)
        for path in protected_paths
    )
    destinations = tuple(
        (_resolved_key(path), path)
        for path in destination_paths
    )
    for destination_key, destination in destinations:
        for other_key, other in (*protected, *destinations):
            if (
                _strict_path_ancestor(destination_key, other_key)
                or _strict_path_ancestor(other_key, destination_key)
            ):
                raise ReferenceMatchContractError(
                    "run file paths must not be nested: "
                    f"{destination} and {other}"
                )


@dataclass(frozen=True)
class _OwnedDirectory:
    path: Path
    directory_identity: tuple[int, int]
    marker_path: Path
    marker_identity: tuple[int, int]
    marker_payload: bytes


def _validate_render_contract(
    protected_input_paths: tuple[Path, ...],
    source_paths: tuple[Path, ...],
    output_paths: tuple[Path, ...],
    artifact_path: Path | None,
    artifact_label: str,
    output_bit_depth: int,
) -> None:
    if (
        len(source_paths) > MAX_REFERENCE_MATCH_BATCH_SOURCES
        or len(output_paths) > MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "file reference match supports at most "
            f"{MAX_REFERENCE_MATCH_BATCH_SOURCES} sources"
        )
    if len(source_paths) != len(output_paths):
        raise ReferenceMatchContractError(
            "source_paths and output_paths must have equal length"
        )
    if output_bit_depth not in {8, 16}:
        raise ReferenceMatchContractError("output_bit_depth must be 8 or 16")
    for protected_input in protected_input_paths:
        if not protected_input.is_file():
            raise ReferenceMatchContractError(
                f"{artifact_label} input must be an existing file"
            )
    for source in source_paths:
        if not source.is_file():
            raise ReferenceMatchContractError(
                f"source path must be an existing file: {source}"
            )
    output_keys = [_resolved_key(path) for path in output_paths]
    if len(set(output_keys)) != len(output_keys):
        raise ReferenceMatchContractError("output paths must be unique")
    protected = {
        *(_resolved_key(path) for path in protected_input_paths),
        *(_resolved_key(path) for path in source_paths),
    }
    if artifact_path is not None:
        if artifact_path.suffix.casefold() != ".json":
            raise ReferenceMatchContractError(
                f"{artifact_label} path must use a .json extension"
            )
        artifact_key = _resolved_key(artifact_path)
        if artifact_key in protected or artifact_key in output_keys:
            raise ReferenceMatchContractError(
                f"{artifact_label} path must not overwrite an input or image output"
            )
        protected.add(artifact_key)
    for output in output_paths:
        if _resolved_key(output) in protected:
            raise ReferenceMatchContractError(
                "output path must not overwrite a reference, source or recipe"
            )
        suffix = output.suffix.casefold()
        allowed = _SDR_OUTPUT_EXTENSIONS if output_bit_depth == 8 else _SDR16_OUTPUT_EXTENSIONS
        if suffix not in allowed:
            raise ReferenceMatchContractError(
                f"unsupported {output_bit_depth}-bit output extension: {suffix or '<none>'}"
            )
    _validate_run_path_topology(
        protected_paths=(*protected_input_paths, *source_paths),
        destination_paths=(
            *output_paths,
            *((artifact_path,) if artifact_path is not None else ()),
        ),
    )


def _validate_file_contract(
    reference_path: Path,
    source_paths: tuple[Path, ...],
    output_paths: tuple[Path, ...],
    recipe_path: Path | None,
    output_bit_depth: int,
) -> None:
    _validate_render_contract(
        (reference_path,),
        source_paths,
        output_paths,
        recipe_path,
        "recipe",
        output_bit_depth,
    )


def _validate_replay_file_contract(
    recipe_path: Path,
    source_paths: tuple[Path, ...],
    output_paths: tuple[Path, ...],
    output_bit_depth: int,
) -> None:
    if recipe_path.suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            "recipe_path must use a .json extension"
        )
    _validate_render_contract(
        (recipe_path,),
        source_paths,
        output_paths,
        None,
        "recipe",
        output_bit_depth,
    )


def _validate_report_contract(
    report_path: Path | None,
    *,
    protected_paths: tuple[Path, ...],
) -> None:
    if report_path is None:
        return
    report_key = _resolved_key(report_path)
    if report_key in {
        _resolved_key(path) for path in protected_paths
    }:
        raise ReferenceMatchContractError(
            "reference-match report must not overwrite a run artifact"
        )
    if report_path.suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            "reference-match report path must use a .json extension"
        )
    if report_path.exists() and report_path.is_dir():
        raise ReferenceMatchContractError(
            "reference-match report path must not be a directory"
        )
    _validate_run_path_topology(
        protected_paths=protected_paths,
        destination_paths=(report_path,),
    )


def _stage_path(destination: Path, token: str) -> Path:
    return destination.with_name(
        f".{destination.stem}.{token}.reference-match-stage{destination.suffix}"
    )


def _backup_path(destination: Path, token: str) -> Path:
    return destination.with_name(
        f".{destination.name}.{token}.reference-match-backup"
    )


def _ensure_parent_directory(
    destination: Path,
    owned_directories: list[_OwnedDirectory],
    *,
    token: str,
) -> None:
    """Create destination parents and bind each to transaction-owned evidence."""

    missing: list[Path] = []
    current = destination.parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise ReferenceMatchContractError(
            f"destination parent is not a directory: {current}"
        )
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise ReferenceMatchContractError(
                    f"destination parent is not a directory: {directory}"
                ) from None
        else:
            directory_identity = _file_identity(directory)
            marker_path = (
                directory
                / f".{token}.reference-match-directory-owner"
            )
            marker_payload = (
                "neuro-film.reference-match-directory-owner.v1\n"
                f"{token}\n"
                f"{directory_identity[0]}:{directory_identity[1]}\n"
            ).encode("ascii")
            with marker_path.open("xb") as marker:
                marker.write(marker_payload)
            owned = _OwnedDirectory(
                path=directory,
                directory_identity=directory_identity,
                marker_path=marker_path,
                marker_identity=_file_identity(marker_path),
                marker_payload=marker_payload,
            )
            owned_directories.append(owned)
            if (
                directory.is_symlink()
                or _file_identity(directory) != directory_identity
            ):
                raise ReferenceMatchContractError(
                    "created destination directory identity changed"
                )


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _move_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one file only when destination is absent."""

    if os.name == "nt":
        # MoveFile without MOVEFILE_REPLACE_EXISTING is atomic on one volume
        # and fails if another writer won the destination name.
        os.rename(source, destination)
        return
    _link_noreplace(source, destination)


def _link_noreplace(source: Path, destination: Path) -> None:
    """Publish with a POSIX hard link; stage cleanup is housekeeping."""

    # POSIX rename replaces an existing destination, so publish by hard-link
    # instead. Stages and destinations share a parent/filesystem.
    os.link(source, destination)
    for _attempt in range(3):
        try:
            source.unlink()
            return
        except OSError:
            continue
    # The destination was already published. Stage cleanup is post-publication
    # housekeeping and must never reclassify a valid final manifest as failed.
    # Do not check-then-unlink destination: that could erase an external winner.


def _file_identity(path: Path) -> tuple[int, int]:
    """Return the stable same-volume identity used for safe rollback."""

    value = path.stat()
    return int(value.st_dev), int(value.st_ino)


def _path_entry_identity(path: Path) -> tuple[int, int]:
    """Return identity for the directory entry itself, without following links."""

    value = path.lstat()
    return int(value.st_dev), int(value.st_ino)


def _path_entry_has_identity(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    try:
        return _path_entry_identity(path) == identity
    except OSError:
        return False


def _remember_owned_file(
    path: Path,
    identities: dict[Path, tuple[int, int]],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ReferenceMatchContractError(
            "transaction staging artifact must be a regular file"
        )
    identities[path] = _path_entry_identity(path)


def _cleanup_owned_files(
    paths: Iterable[Path],
    identities: Mapping[Path, tuple[int, int]],
) -> None:
    """Unlink only staging entries whose exact identity is transaction-owned."""

    for path in paths:
        identity = identities.get(path)
        if (
            identity is None
            or not _path_entry_has_identity(path, identity)
        ):
            continue
        try:
            path.unlink()
        except OSError:
            pass


def _owned_directory_is_current(owned: _OwnedDirectory) -> bool:
    try:
        return (
            not owned.path.is_symlink()
            and _file_identity(owned.path) == owned.directory_identity
            and not owned.marker_path.is_symlink()
            and _file_identity(owned.marker_path) == owned.marker_identity
            and owned.marker_path.read_bytes() == owned.marker_payload
        )
    except OSError:
        return False


def _cleanup_owned_directories(
    owned_directories: list[_OwnedDirectory],
    *,
    remove_directories: bool,
) -> None:
    """Remove only marker-proven directory state owned by this transaction."""

    for owned in reversed(owned_directories):
        if not _owned_directory_is_current(owned):
            continue
        try:
            owned.marker_path.unlink()
        except OSError:
            continue
        if not remove_directories:
            continue
        try:
            # Revalidate after marker removal. If another process replaced the
            # path, preserving the new directory is safer than path-based
            # cleanup. A subsequent non-empty change also makes rmdir fail.
            if (
                owned.path.is_symlink()
                or _file_identity(owned.path)
                != owned.directory_identity
            ):
                continue
            owned.path.rmdir()
        except OSError:
            pass


def _commit_staged_batch_unlocked(
    pairs: tuple[tuple[Path, Path], ...],
    *,
    token: str,
    cleanup: list[Path],
    expected_stage_sha256: Mapping[Path, str] | None = None,
    expected_destination_sha256: Mapping[Path, str | None] | None = None,
    replace_existing: bool = True,
) -> None:
    """Publish staged files under replacement or manifest-last semantics.

    The default preserves the historical replacement behavior. Callers may
    instead select create-only publication, which rejects all prior hashes and
    uses an atomic no-replace operation for each destination. Create-only
    callers must place their commit manifest last; failures can leave earlier
    files as uncommitted orphans because deleting them would risk erasing an
    external replacement. Once every publication succeeds, the new transaction
    is committed. Removing old backup files is post-commit housekeeping and
    must not turn a successful commit into a reported failure. Transient
    cleanup errors are retried; a persistently undeletable backup is retained
    for recovery.
    """

    stages = {stage for stage, _destination in pairs}
    destinations = {destination for _stage, destination in pairs}
    if (
        expected_stage_sha256 is not None
        and set(expected_stage_sha256) != stages
    ):
        raise ReferenceMatchContractError(
            "batch stage hash inventory does not match commit pairs"
        )
    if (
        expected_destination_sha256 is not None
        and set(expected_destination_sha256) != destinations
    ):
        raise ReferenceMatchContractError(
            "batch destination hash inventory does not match commit pairs"
        )
    if (
        not replace_existing
        and expected_destination_sha256 is not None
        and any(
            value is not None
            for value in expected_destination_sha256.values()
        )
    ):
        raise ReferenceMatchContractError(
            "create-only batch commit cannot replace prior destinations"
        )

    def checked_hash(value: str, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ReferenceMatchContractError(
                f"{label} must be a lowercase SHA-256"
            )
        return value

    committed: list[
        tuple[
            Path,
            Path | None,
            tuple[int, int],
            tuple[int, int] | None,
            str | None,
        ]
    ] = []
    try:
        for stage, destination in pairs:
            if not stage.is_file() or stage.is_symlink():
                raise ReferenceMatchContractError(
                    "batch stage must be a regular file"
                )
            destination_exists = os.path.lexists(destination)
            if destination_exists and (
                destination.is_symlink() or not destination.is_file()
            ):
                raise ReferenceMatchContractError(
                    "batch destination must be a regular file when it exists"
                )
            published_identity = _path_entry_identity(stage)
            published_sha256: str | None = None
            if expected_stage_sha256 is not None:
                expected_stage = checked_hash(
                    expected_stage_sha256[stage],
                    "expected stage hash",
                )
                if (
                    not stage.is_file()
                    or sha256_file(stage) != expected_stage
                ):
                    raise ReferenceMatchContractError(
                        "staged file changed before batch replacement"
                    )
                published_sha256 = expected_stage
            if expected_destination_sha256 is not None:
                expected_destination = expected_destination_sha256[
                    destination
                ]
                if expected_destination is None:
                    if os.path.lexists(destination):
                        raise ReferenceMatchContractError(
                            "destination appeared before batch replacement"
                        )
                else:
                    checked_hash(
                        expected_destination,
                        "expected destination hash",
                    )
                    if (
                        not destination.is_file()
                        or sha256_file(destination)
                        != expected_destination
                    ):
                        raise ReferenceMatchContractError(
                            "destination changed before batch replacement"
                        )
            backup: Path | None = None
            backup_identity: tuple[int, int] | None = None
            if replace_existing and destination_exists:
                original_identity = _path_entry_identity(destination)
                backup = _backup_path(destination, token)
                _replace(destination, backup)
                cleanup.append(backup)
                backup_identity = _path_entry_identity(backup)
                if backup_identity != original_identity:
                    cleanup.remove(backup)
                    raise ReferenceMatchContractError(
                        "destination changed while creating rollback backup"
                    )
            try:
                if replace_existing:
                    _replace(stage, destination)
                else:
                    _move_noreplace(stage, destination)
            except Exception:
                if backup is not None:
                    if (
                        backup_identity is None
                        or not _path_entry_has_identity(
                            backup,
                            backup_identity,
                        )
                        or os.path.lexists(destination)
                    ):
                        if backup in cleanup:
                            cleanup.remove(backup)
                        raise ReferenceMatchContractError(
                            "batch publish failed and rollback backup "
                            "could not be restored safely"
                        ) from None
                    try:
                        _move_noreplace(backup, destination)
                    except OSError as exc:
                        if backup in cleanup:
                            cleanup.remove(backup)
                        raise ReferenceMatchContractError(
                            "batch publish failed and rollback backup "
                            "could not be restored safely"
                        ) from exc
                    cleanup.remove(backup)
                raise
            committed.append(
                (
                    destination,
                    backup,
                    published_identity,
                    backup_identity,
                    published_sha256,
                )
            )
            cleanup.remove(stage)
            if (
                not destination.is_file()
                or destination.is_symlink()
                or not _path_entry_has_identity(
                    destination,
                    published_identity,
                )
            ):
                raise ReferenceMatchContractError(
                    "destination changed during publication"
                )
            if (
                published_sha256 is not None
                and sha256_file(destination) != published_sha256
            ):
                raise ReferenceMatchContractError(
                    "committed destination bytes differ from staged hash"
                )
        if expected_stage_sha256 is not None and replace_existing:
            for (
                (stage, destination),
                (
                    committed_destination,
                    _backup,
                    published_identity,
                    _backup_identity,
                    _published_sha256,
                ),
            ) in zip(pairs, committed, strict=True):
                if committed_destination != destination:
                    raise ReferenceMatchContractError(
                        "batch commit inventory order changed"
                    )
                expected = expected_stage_sha256[stage]
                if (
                    not destination.is_file()
                    or sha256_file(destination) != expected
                ):
                    raise ReferenceMatchContractError(
                        "committed destination bytes differ from staged hash"
                    )
                if (
                    destination.is_symlink()
                    or not _path_entry_has_identity(
                        destination,
                        published_identity,
                    )
                ):
                    raise ReferenceMatchContractError(
                        "committed destination identity changed"
                    )
    except Exception:
        if not replace_existing:
            # Create-only callers publish their commit manifest last. Earlier
            # files are immutable, uncommitted orphans when a later publish
            # fails. Deleting them here would introduce an unavoidable
            # check-then-unlink race that could erase a non-cooperating
            # writer's replacement. A verifier/GC may handle proven orphans.
            raise
        rollback_errors: list[str] = []
        for (
            destination,
            backup,
            published_identity,
            backup_identity,
            _published_sha256,
        ) in reversed(committed):
            try:
                if os.path.lexists(destination):
                    if not _path_entry_has_identity(
                        destination,
                        published_identity,
                    ):
                        if backup is not None and backup in cleanup:
                            cleanup.remove(backup)
                        rollback_errors.append(
                            f"{destination}: published destination was replaced"
                        )
                        continue
                    destination.unlink()
                if backup is not None:
                    if (
                        backup_identity is None
                        or not _path_entry_has_identity(
                            backup,
                            backup_identity,
                        )
                    ):
                        if backup in cleanup:
                            cleanup.remove(backup)
                        rollback_errors.append(
                            f"{destination}: rollback backup was replaced"
                        )
                        continue
                    _move_noreplace(backup, destination)
                    cleanup.remove(backup)
            except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
                if backup is not None and backup in cleanup:
                    cleanup.remove(backup)
                rollback_errors.append(f"{destination}: {exc}")
        if rollback_errors:
            raise ReferenceMatchContractError(
                "batch commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    for (
        _destination,
        backup,
        _published_identity,
        backup_identity,
        _sha256,
    ) in committed:
        if backup is None:
            continue
        if (
            backup_identity is not None
            and _path_entry_has_identity(backup, backup_identity)
        ):
            for _attempt in range(3):
                try:
                    backup.unlink(missing_ok=True)
                    break
                except OSError:
                    continue
        if backup in cleanup:
            # Do not let the caller's finally block reclassify an already
            # committed transaction. A persistent backup is recoverable
            # debris, never proof that the new destinations were rolled back.
            cleanup.remove(backup)


def _commit_staged_batch(
    pairs: tuple[tuple[Path, Path], ...],
    *,
    token: str,
    cleanup: list[Path],
    expected_stage_sha256: Mapping[Path, str] | None = None,
    expected_destination_sha256: Mapping[Path, str | None] | None = None,
    replace_existing: bool = True,
    targets_already_locked: bool = False,
) -> None:
    if targets_already_locked:
        _commit_staged_batch_unlocked(
            pairs,
            token=token,
            cleanup=cleanup,
            expected_stage_sha256=expected_stage_sha256,
            expected_destination_sha256=expected_destination_sha256,
            replace_existing=replace_existing,
        )
        return
    with target_transaction_lock(
        tuple(destination for _stage, destination in pairs)
    ):
        _commit_staged_batch_unlocked(
            pairs,
            token=token,
            cleanup=cleanup,
            expected_stage_sha256=expected_stage_sha256,
            expected_destination_sha256=expected_destination_sha256,
            replace_existing=replace_existing,
        )


def _encode_srgb(
    pixels: np.ndarray,
    destination: Path,
    *,
    output_bit_depth: int,
) -> tuple[str, float]:
    clipped = np.any((pixels < 0.0) | (pixels > 1.0), axis=-1)
    if float(np.min(pixels)) < -2e-6 or float(np.max(pixels)) > 1.0 + 2e-6:
        raise ReferenceMatchContractError(
            "render output exceeds the bounded sRGB encoding tolerance"
        )
    if output_bit_depth == 8:
        output_format = save_srgb8(pixels, destination)
    elif destination.suffix.casefold() == ".png":
        output_format = save_srgb16_png(pixels, destination)
    else:
        output_format = save_srgb16_tiff(pixels, destination)
    return output_format, float(np.mean(clipped, dtype=np.float64))


def _encode_working_image(
    image: WorkingImage,
    destination: Path,
    *,
    output_bit_depth: int,
) -> tuple[str, float]:
    resolve_reference_file_output_capability(
        working_space=image.working_space,
        transfer_state=image.transfer_state,
        output_bit_depth=output_bit_depth,
        output_extension=destination.suffix,
    )
    if image.working_space == "linear_srgb":
        encoded = working_image_to_srgb_float(image)
        return _encode_srgb(
            encoded,
            destination,
            output_bit_depth=output_bit_depth,
        )
    if image.working_space != "linear_rec2020":
        raise ReferenceMatchContractError(
            "file adapter output working space is unsupported"
        )
    pixels = image.pixels
    clipped = np.any((pixels < 0.0) | (pixels > 1.0), axis=-1)
    if float(np.min(pixels)) < -2e-6 or float(np.max(pixels)) > 1.0 + 2e-6:
        raise ReferenceMatchContractError(
            "render output exceeds the bounded Rec.2020 encoding tolerance"
        )
    save_rec2020_16_png(image, destination)
    return "PNG", float(np.mean(clipped, dtype=np.float64))


def _load_stable_working_image(
    path: Path,
    *,
    label: str,
) -> tuple[WorkingImage, str]:
    """Decode one input and bind it to stable bytes at the path boundary."""

    before = sha256_file(path)
    image = load_working_image(path)
    after = sha256_file(path)
    if after != before:
        raise ReferenceMatchContractError(
            f"{label} file changed while it was being decoded"
        )
    return image, before


def _execute_file_render(
    recipe: ReferenceLookRecipe,
    sources: tuple[Path, ...],
    outputs: tuple[Path, ...],
    *,
    guard_policy: ReferenceRenderGuardPolicy | None,
    output_bit_depth: int,
    recipe_destination: Path | None = None,
    report_destination: Path | None = None,
    report_payload_factory: _ReportPayloadFactory | None = None,
) -> tuple[
    tuple[FileReferenceMatchOutput, ...],
    str | None,
    str | None,
]:
    """Commit a complete image/recipe/report run as one transaction."""

    if (report_destination is None) != (report_payload_factory is None):
        raise ReferenceMatchContractError(
            "report destination and payload factory must be supplied together"
        )

    token = uuid.uuid4().hex
    staged: list[Path] = []
    staged_outputs: list[
        tuple[
            Path,
            str,
            Path,
            str,
            float,
            ReferenceMatchDiagnostics,
            ReferenceSafetyDecision,
        ]
    ] = []
    staged_recipe: Path | None = None
    staged_report: Path | None = None
    staged_identities: dict[Path, tuple[int, int]] = {}
    owned_directories: list[_OwnedDirectory] = []
    committed = False
    try:
        if recipe_destination is not None:
            _ensure_parent_directory(
                recipe_destination,
                owned_directories,
                token=token,
            )
            staged_recipe = _stage_path(recipe_destination, token)
            staged.append(staged_recipe)
            save_reference_look_recipe(recipe, staged_recipe)
            _remember_owned_file(staged_recipe, staged_identities)

        for index, (source_path, output_path) in enumerate(
            zip(sources, outputs, strict=True)
        ):
            source, source_file_sha256 = _load_stable_working_image(
                source_path,
                label="source",
            )
            if (
                source.working_space,
                source.transfer_state,
            ) not in _SUPPORTED_FILE_INPUT_RAILS:
                raise ReferenceMatchContractError(
                    "file adapter currently requires display-linear "
                    "linear_srgb or linear_rec2020 sources"
                )
            rendered = render_reference_look_guarded(
                recipe,
                source,
                source_index=index,
                policy=guard_policy,
            )
            # Guarded rendering returns a distinct output WorkingImage and
            # diagnostics. The decoded source is no longer needed while that
            # output is encoded, so do not retain both full-resolution arrays.
            del source
            _ensure_parent_directory(
                output_path,
                owned_directories,
                token=token,
            )
            stage = _stage_path(output_path, token)
            staged.append(stage)
            output_format, clipped_fraction = _encode_working_image(
                rendered.image,
                stage,
                output_bit_depth=output_bit_depth,
            )
            _remember_owned_file(stage, staged_identities)
            output_capability = resolve_reference_file_output_capability(
                working_space=rendered.image.working_space,
                transfer_state=rendered.image.transfer_state,
                output_bit_depth=output_bit_depth,
                output_extension=stage.suffix,
            )
            metadata_attestation = attest_reference_file_output_metadata(
                stage,
                encoding_profile=output_capability.encoding_profile,
            )
            if not metadata_attestation.accepted:
                raise ReferenceMatchContractError(
                    "rendered output violates the metadata-minimization "
                    f"policy: {metadata_attestation.failure_code}"
                )
            staged_outputs.append(
                (
                    source_path,
                    source_file_sha256,
                    output_path,
                    output_format,
                    clipped_fraction,
                    rendered.candidate_diagnostics,
                    rendered.safety,
                )
            )
            # Only scalar metadata and immutable diagnostics are retained.
            # Release the encoded WorkingImage before the next source load.
            del rendered

        prepared = tuple(
            FileReferenceMatchOutput(
                source_path=source_path,
                source_file_sha256=source_file_sha256,
                output_path=output_path,
                output_sha256=sha256_file(
                    _stage_path(output_path, token)
                ),
                output_format=output_format,
                output_bit_depth=output_bit_depth,
                encode_clipped_fraction=clipped_fraction,
                diagnostics=diagnostics,
                safety=safety,
            )
            for (
                source_path,
                source_file_sha256,
                output_path,
                output_format,
                clipped_fraction,
                diagnostics,
                safety,
            ) in staged_outputs
        )
        recipe_file_sha256 = (
            None
            if staged_recipe is None
            else sha256_file(staged_recipe)
        )
        commit_pairs = [
            (_stage_path(output_path, token), output_path)
            for (
                _source_path,
                _source_file_sha256,
                output_path,
                _format,
                _clipped,
                _diagnostics,
                _safety,
            )
            in staged_outputs
        ]
        if recipe_destination is not None and staged_recipe is not None:
            commit_pairs.append((staged_recipe, recipe_destination))
        report_file_sha256: str | None = None
        if (
            report_destination is not None
            and report_payload_factory is not None
        ):
            _ensure_parent_directory(
                report_destination,
                owned_directories,
                token=token,
            )
            staged_report = _stage_path(report_destination, token)
            staged.append(staged_report)
            payload = report_payload_factory(
                prepared,
                recipe_file_sha256,
            )
            if not isinstance(payload, Mapping):
                raise ReferenceMatchContractError(
                    "report payload factory must return an object"
                )
            atomic_write_json(staged_report, payload)
            _remember_owned_file(staged_report, staged_identities)
            report_file_sha256 = sha256_file(staged_report)
            commit_pairs.append((staged_report, report_destination))
        expected_stage_sha256 = {
            stage: sha256_file(stage)
            for stage, _destination in commit_pairs
        }
        _commit_staged_batch(
            tuple(commit_pairs),
            token=token,
            cleanup=staged,
            expected_stage_sha256=expected_stage_sha256,
        )
        committed = True
        return prepared, recipe_file_sha256, report_file_sha256
    finally:
        _cleanup_owned_files(staged, staged_identities)
        _cleanup_owned_directories(
            owned_directories,
            remove_directories=not committed,
        )


def match_reference_files(
    reference_path: Path | str,
    source_paths: Iterable[Path | str],
    output_paths: Iterable[Path | str],
    *,
    recipe_path: Path | str | None = None,
    report_path: Path | str | None = None,
    policy: ReferenceLookPolicy | None = None,
    guard_policy: ReferenceRenderGuardPolicy | None = None,
    output_bit_depth: int = 16,
) -> FileReferenceMatchResult:
    """Fit once and transactionally render one uploaded reference across N files."""

    reference = Path(reference_path)
    sources = _paths(source_paths, "source_paths")
    outputs = _paths(output_paths, "output_paths")
    recipe_destination = Path(recipe_path) if recipe_path is not None else None
    report_destination = Path(report_path) if report_path is not None else None
    _validate_file_contract(
        reference,
        sources,
        outputs,
        recipe_destination,
        output_bit_depth,
    )
    _validate_report_contract(
        report_destination,
        protected_paths=(
            reference,
            *sources,
            *outputs,
            *((recipe_destination,) if recipe_destination is not None else ()),
        ),
    )

    reference_working, reference_file_sha256 = _load_stable_working_image(
        reference,
        label="reference",
    )
    recipe = fit_reference_look(reference_working, policy=policy)
    # The recipe owns only fitted statistics and exact reference identities.
    # Release the decoded pixels before the first source is decoded so large
    # reference and source WorkingImages do not overlap for the whole batch.
    del reference_working

    def report_factory(
        prepared: tuple[FileReferenceMatchOutput, ...],
        recipe_file_sha256: str | None,
    ) -> Mapping[str, Any]:
        from .reporting import build_file_match_report

        return build_file_match_report(
            FileReferenceMatchResult(
                reference_path=reference,
                reference_file_sha256=reference_file_sha256,
                recipe=recipe,
                recipe_path=recipe_destination,
                recipe_file_sha256=recipe_file_sha256,
                outputs=prepared,
                report_path=report_destination,
            )
        )

    committed, recipe_file_sha256, report_file_sha256 = _execute_file_render(
        recipe,
        sources,
        outputs,
        guard_policy=guard_policy,
        output_bit_depth=output_bit_depth,
        recipe_destination=recipe_destination,
        report_destination=report_destination,
        report_payload_factory=(
            report_factory if report_destination is not None else None
        ),
    )
    return FileReferenceMatchResult(
        reference_path=reference,
        reference_file_sha256=reference_file_sha256,
        recipe=recipe,
        recipe_path=recipe_destination,
        recipe_file_sha256=recipe_file_sha256,
        outputs=committed,
        report_path=report_destination,
        report_file_sha256=report_file_sha256,
    )


def replay_reference_files(
    recipe_path: Path | str,
    source_paths: Iterable[Path | str],
    output_paths: Iterable[Path | str],
    *,
    report_path: Path | str | None = None,
    guard_policy: ReferenceRenderGuardPolicy | None = None,
    output_bit_depth: int = 16,
) -> FileReferenceReplayResult:
    """Transactionally replay one verified stored recipe across N files."""

    recipe_source = Path(recipe_path)
    sources = _paths(source_paths, "source_paths")
    outputs = _paths(output_paths, "output_paths")
    report_destination = Path(report_path) if report_path is not None else None
    _validate_replay_file_contract(
        recipe_source,
        sources,
        outputs,
        output_bit_depth,
    )
    _validate_report_contract(
        report_destination,
        protected_paths=(recipe_source, *sources, *outputs),
    )
    recipe, recipe_file_sha256 = load_reference_look_recipe_bound(
        recipe_source
    )

    def report_factory(
        prepared: tuple[FileReferenceMatchOutput, ...],
        _persisted_recipe_sha256: str | None,
    ) -> Mapping[str, Any]:
        from .reporting import build_file_replay_report

        return build_file_replay_report(
            FileReferenceReplayResult(
                recipe=recipe,
                recipe_path=recipe_source,
                recipe_file_sha256=recipe_file_sha256,
                outputs=prepared,
                report_path=report_destination,
            )
        )

    (
        committed,
        persisted_recipe_sha256,
        report_file_sha256,
    ) = _execute_file_render(
        recipe,
        sources,
        outputs,
        guard_policy=guard_policy,
        output_bit_depth=output_bit_depth,
        report_destination=report_destination,
        report_payload_factory=(
            report_factory if report_destination is not None else None
        ),
    )
    if persisted_recipe_sha256 is not None:  # pragma: no cover - invariant.
        raise AssertionError("recipe replay must not persist a second recipe")
    return FileReferenceReplayResult(
        recipe=recipe,
        recipe_path=recipe_source,
        recipe_file_sha256=recipe_file_sha256,
        outputs=committed,
        report_path=report_destination,
        report_file_sha256=report_file_sha256,
    )


__all__ = [
    "REFERENCE_FILE_INPUT_INSPECTION_CLAIM_CEILING",
    "REFERENCE_FILE_INPUT_INSPECTION_BATCH_SCHEMA_ID",
    "REFERENCE_FILE_INPUT_INSPECTION_SCHEMA_ID",
    "REFERENCE_FILE_OUTPUT_CAPABILITIES_ID",
    "FileReferenceInputInspection",
    "FileReferenceOutputCapability",
    "FileReferenceMatchOutput",
    "FileReferenceMatchResult",
    "FileReferenceReplayResult",
    "inspect_reference_file_input",
    "inspect_reference_file_inputs",
    "match_reference_files",
    "reference_file_supported_input_rails",
    "reference_file_output_capabilities",
    "reference_file_output_capabilities_payload",
    "resolve_reference_file_output_capability",
    "replay_reference_files",
]
