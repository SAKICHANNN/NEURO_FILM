"""Deterministic provenance report for one file-level reference-match run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.inference.render_contract import atomic_write_json, sha256_file

from .batch_limits import MAX_REFERENCE_MATCH_BATCH_SOURCES
from .contracts import ReferenceMatchContractError
from .files import (
    FileReferenceMatchOutput,
    FileReferenceMatchResult,
    FileReferenceReplayResult,
)


REFERENCE_MATCH_REPORT_SCHEMA_ID = "neuro-film.reference-match-report.v1"
REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID = (
    "neuro-film.reference-match-replay-report.v1"
)


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _output_rows(
    rows: tuple[FileReferenceMatchOutput, ...],
) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, tuple)
        or not rows
        or len(rows) > MAX_REFERENCE_MATCH_BATCH_SOURCES
    ):
        raise ReferenceMatchContractError(
            "reference-match report outputs must contain between 1 and "
            f"{MAX_REFERENCE_MATCH_BATCH_SOURCES} rows"
        )
    outputs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, FileReferenceMatchOutput):
            raise ReferenceMatchContractError(
                "reference-match report output row type is invalid"
            )
        diagnostics = asdict(row.diagnostics)
        diagnostics["source_shape"] = list(diagnostics["source_shape"])
        safety = asdict(row.safety)
        safety["reasons"] = list(safety["reasons"])
        outputs.append(
            {
                "source_path": str(row.source_path.resolve()),
                "source_sha256": _sha256(
                    row.source_file_sha256,
                    "source_file_sha256",
                ),
                "output_path": str(row.output_path.resolve()),
                "output_sha256": row.output_sha256,
                "output_format": row.output_format,
                "output_bit_depth": row.output_bit_depth,
                "encode_clipped_fraction": row.encode_clipped_fraction,
                "candidate_diagnostics": diagnostics,
                "safety": safety,
            }
        )
    return outputs


def build_file_match_report(
    result: FileReferenceMatchResult,
) -> dict[str, Any]:
    """Build a JSON-safe report covering every input, output and guard decision."""

    if not isinstance(result, FileReferenceMatchResult):
        raise ReferenceMatchContractError(
            "result must be FileReferenceMatchResult"
        )
    report: dict[str, Any] = {
        "schema_id": REFERENCE_MATCH_REPORT_SCHEMA_ID,
        "algorithm_id": result.recipe.algorithm_id,
        "recipe_id": result.recipe.recipe_id,
        "claim_ceiling": result.recipe.claim_ceiling,
        "evidence_grade": result.recipe.evidence_grade,
        "reference": {
            "path": str(result.reference_path.resolve()),
            "file_sha256": _sha256(
                result.reference_file_sha256,
                "reference_file_sha256",
            ),
            "pixel_sha256": result.recipe.reference_pixel_sha256,
        },
        "recipe_file": (
            None
            if result.recipe_path is None
            else {
                "path": str(result.recipe_path.resolve()),
                "sha256": result.recipe_file_sha256,
            }
        ),
        "outputs": _output_rows(result.outputs),
    }
    return report


def build_file_replay_report(
    result: FileReferenceReplayResult,
) -> dict[str, Any]:
    """Build provenance for a recipe-only replay without inventing a reference."""

    if not isinstance(result, FileReferenceReplayResult):
        raise ReferenceMatchContractError(
            "result must be FileReferenceReplayResult"
        )
    return {
        "schema_id": REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID,
        "operation": "recipe-replay",
        "algorithm_id": result.recipe.algorithm_id,
        "recipe_id": result.recipe.recipe_id,
        "claim_ceiling": result.recipe.claim_ceiling,
        "evidence_grade": result.recipe.evidence_grade,
        "reference_pixel_sha256": result.recipe.reference_pixel_sha256,
        "recipe_file": {
            "path": str(result.recipe_path.resolve()),
            "sha256": result.recipe_file_sha256,
        },
        "outputs": _output_rows(result.outputs),
    }


def _resolved_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def save_file_match_report(
    result: FileReferenceMatchResult,
    path: Path | str,
) -> str:
    """Atomically persist a report without overwriting any run artifact."""

    destination = Path(path)
    protected = {
        _resolved_key(result.reference_path),
        *(_resolved_key(row.source_path) for row in result.outputs),
        *(_resolved_key(row.output_path) for row in result.outputs),
    }
    if result.recipe_path is not None:
        protected.add(_resolved_key(result.recipe_path))
    if _resolved_key(destination) in protected:
        raise ReferenceMatchContractError(
            "reference-match report must not overwrite a run input or output"
        )
    if destination.suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            "reference-match report path must use a .json extension"
        )
    if destination.exists() and destination.is_dir():
        raise ReferenceMatchContractError(
            "reference-match report path must not be a directory"
        )
    atomic_write_json(destination, build_file_match_report(result))
    return sha256_file(destination)


def save_file_replay_report(
    result: FileReferenceReplayResult,
    path: Path | str,
) -> str:
    """Atomically save replay provenance without overwriting replay artifacts."""

    destination = Path(path)
    protected = {
        _resolved_key(result.recipe_path),
        *(_resolved_key(row.source_path) for row in result.outputs),
        *(_resolved_key(row.output_path) for row in result.outputs),
    }
    if _resolved_key(destination) in protected:
        raise ReferenceMatchContractError(
            "reference-match replay report must not overwrite a run artifact"
        )
    if destination.suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            "reference-match replay report path must use a .json extension"
        )
    if destination.exists() and destination.is_dir():
        raise ReferenceMatchContractError(
            "reference-match replay report path must not be a directory"
        )
    atomic_write_json(destination, build_file_replay_report(result))
    return sha256_file(destination)


__all__ = [
    "REFERENCE_MATCH_REPORT_SCHEMA_ID",
    "REFERENCE_MATCH_REPLAY_REPORT_SCHEMA_ID",
    "build_file_match_report",
    "build_file_replay_report",
    "save_file_match_report",
    "save_file_replay_report",
]
