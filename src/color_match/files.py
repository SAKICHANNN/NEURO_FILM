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


def _stage_path(destination: Path, token: str) -> Path:
    return destination.with_name(
        f".{destination.stem}.{token}.reference-match-stage{destination.suffix}"
    )


def _backup_path(destination: Path, token: str) -> Path:
    return destination.with_name(
        f".{destination.name}.{token}.reference-match-backup"
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
        tuple[Path, Path | None, tuple[int, int] | None, str | None]
    ] = []
    try:
        for stage, destination in pairs:
            published_identity: tuple[int, int] | None = None
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
            if replace_existing and destination.exists():
                backup = _backup_path(destination, token)
                _replace(destination, backup)
                cleanup.append(backup)
            try:
                if replace_existing:
                    _replace(stage, destination)
                else:
                    published_identity = _file_identity(stage)
                    _move_noreplace(stage, destination)
            except Exception:
                if backup is not None and backup.exists():
                    _replace(backup, destination)
                    cleanup.remove(backup)
                raise
            if (
                not replace_existing
                and (
                    not destination.is_file()
                    or destination.is_symlink()
                    or _file_identity(destination)
                    != published_identity
                    or (
                        published_sha256 is not None
                        and sha256_file(destination)
                        != published_sha256
                    )
                )
            ):
                raise ReferenceMatchContractError(
                    "create-only destination changed during publication"
                )
            committed.append(
                (
                    destination,
                    backup,
                    published_identity,
                    published_sha256,
                )
            )
            cleanup.remove(stage)
        if expected_stage_sha256 is not None and replace_existing:
            for (
                (stage, destination),
                (
                    committed_destination,
                    _backup,
                    published_identity,
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
                    published_identity is not None
                    and (
                        destination.is_symlink()
                        or _file_identity(destination)
                        != published_identity
                    )
                ):
                    raise ReferenceMatchContractError(
                        "committed create-only destination identity changed"
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
            _published_identity,
            _published_sha256,
        ) in reversed(committed):
            try:
                destination.unlink(missing_ok=True)
                if (
                    backup is not None
                    and backup.exists()
                ):
                    _replace(backup, destination)
                    cleanup.remove(backup)
            except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
                rollback_errors.append(f"{destination}: {exc}")
        if rollback_errors:
            raise ReferenceMatchContractError(
                "batch commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    for _destination, backup, _identity, _sha256 in committed:
        if backup is None:
            continue
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
    if output_bit_depth != 16 or destination.suffix.casefold() != ".png":
        raise ReferenceMatchContractError(
            "linear_rec2020 file output requires 16-bit PNG"
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
    try:
        if recipe_destination is not None:
            recipe_destination.parent.mkdir(parents=True, exist_ok=True)
            staged_recipe = _stage_path(recipe_destination, token)
            staged.append(staged_recipe)
            save_reference_look_recipe(recipe, staged_recipe)

        for index, (source_path, output_path) in enumerate(
            zip(sources, outputs, strict=True)
        ):
            source, source_file_sha256 = _load_stable_working_image(
                source_path,
                label="source",
            )
            if (
                source.working_space
                not in {"linear_srgb", "linear_rec2020"}
                or source.transfer_state != "display_linear"
            ):
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output_path, token)
            staged.append(stage)
            output_format, clipped_fraction = _encode_working_image(
                rendered.image,
                stage,
                output_bit_depth=output_bit_depth,
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
            report_destination.parent.mkdir(parents=True, exist_ok=True)
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
        return prepared, recipe_file_sha256, report_file_sha256
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


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
    "FileReferenceMatchOutput",
    "FileReferenceMatchResult",
    "FileReferenceReplayResult",
    "match_reference_files",
    "replay_reference_files",
]
