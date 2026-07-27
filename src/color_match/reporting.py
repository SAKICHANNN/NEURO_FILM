"""Deterministic provenance report for one file-level reference-match run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.inference.render_contract import atomic_write_json, sha256_file

from .contracts import ReferenceMatchContractError
from .files import FileReferenceMatchResult


REFERENCE_MATCH_REPORT_SCHEMA_ID = "neuro-film.reference-match-report.v1"


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
            "file_sha256": sha256_file(result.reference_path),
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
        "outputs": [],
    }
    outputs: list[dict[str, Any]] = report["outputs"]
    for row in result.outputs:
        diagnostics = asdict(row.diagnostics)
        diagnostics["source_shape"] = list(diagnostics["source_shape"])
        safety = asdict(row.safety)
        safety["reasons"] = list(safety["reasons"])
        outputs.append(
            {
                "source_path": str(row.source_path.resolve()),
                "source_sha256": sha256_file(row.source_path),
                "output_path": str(row.output_path.resolve()),
                "output_sha256": row.output_sha256,
                "output_format": row.output_format,
                "output_bit_depth": row.output_bit_depth,
                "encode_clipped_fraction": row.encode_clipped_fraction,
                "candidate_diagnostics": diagnostics,
                "safety": safety,
            }
        )
    return report


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


__all__ = [
    "REFERENCE_MATCH_REPORT_SCHEMA_ID",
    "build_file_match_report",
    "save_file_match_report",
]
