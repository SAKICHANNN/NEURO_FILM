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
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    working_image_to_srgb_float,
)

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


_SDR_OUTPUT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
_SDR16_OUTPUT_EXTENSIONS = frozenset({".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class FileReferenceMatchOutput:
    """Durable output identity and render diagnostics for one source file."""

    source_path: Path
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
    try:
        paths = tuple(Path(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            f"{label} must be an iterable of paths"
        ) from exc
    if not paths:
        raise ReferenceMatchContractError(f"{label} must not be empty")
    return paths


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


def _commit_staged_batch(
    pairs: tuple[tuple[Path, Path], ...],
    *,
    token: str,
    cleanup: list[Path],
) -> None:
    """Commit staged files and restore every prior destination on failure."""

    committed: list[tuple[Path, Path | None]] = []
    try:
        for stage, destination in pairs:
            backup: Path | None = None
            if destination.exists():
                backup = _backup_path(destination, token)
                _replace(destination, backup)
                cleanup.append(backup)
            try:
                _replace(stage, destination)
            except Exception:
                if backup is not None and backup.exists():
                    _replace(backup, destination)
                    cleanup.remove(backup)
                raise
            cleanup.remove(stage)
            committed.append((destination, backup))
    except Exception:
        rollback_errors: list[str] = []
        for destination, backup in reversed(committed):
            try:
                destination.unlink(missing_ok=True)
                if backup is not None and backup.exists():
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
    for _destination, backup in committed:
        if backup is not None:
            backup.unlink(missing_ok=True)
            cleanup.remove(backup)


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
            source = load_working_image(source_path)
            if (
                source.working_space != "linear_srgb"
                or source.transfer_state != "display_linear"
            ):
                raise ReferenceMatchContractError(
                    "file adapter currently requires display-linear "
                    "linear_srgb sources"
                )
            rendered = render_reference_look_guarded(
                recipe,
                source,
                source_index=index,
                policy=guard_policy,
            )
            encoded = working_image_to_srgb_float(rendered.image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stage = _stage_path(output_path, token)
            staged.append(stage)
            output_format, clipped_fraction = _encode_srgb(
                encoded,
                stage,
                output_bit_depth=output_bit_depth,
            )
            staged_outputs.append(
                (
                    source_path,
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
        _commit_staged_batch(
            tuple(commit_pairs),
            token=token,
            cleanup=staged,
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

    reference_working = load_working_image(reference)
    recipe = fit_reference_look(reference_working, policy=policy)

    def report_factory(
        prepared: tuple[FileReferenceMatchOutput, ...],
        recipe_file_sha256: str | None,
    ) -> Mapping[str, Any]:
        from .reporting import build_file_match_report

        return build_file_match_report(
            FileReferenceMatchResult(
                reference_path=reference,
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
