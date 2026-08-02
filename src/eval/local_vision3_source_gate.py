"""U5.R2BU5 metadata-only gate for the legacy local VISION3 raster corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u5_r2bu5_local_vision3_source_gate_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu5_local_vision3_source_gate_report.v1"
STYLES = ("vision3_50d", "vision3_250d", "vision3_500t")
REQUIRED_FIELDS = (
    "path",
    "source",
    "license",
    "redistributable",
    "sha256",
    "author",
    "source_url",
    "film_stock_id",
    "roll_id",
    "process_id",
    "scanner_id",
)


class LocalVision3SourceGateError(RuntimeError):
    """Raised when the frozen metadata-only BU5 contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise LocalVision3SourceGateError("BU5 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU5"
        or payload.get("manifest", {}).get("sha256")
        != "0a58ef7775ff84d05f5744f42b3bc5183281590043bb35de58059b4fd73df233"
        or payload.get("manifest", {}).get("bytes") != 1766345
        or tuple(payload.get("accepted_style_ids", ())) != STYLES
        or tuple(payload.get("required_fields", ())) != REQUIRED_FIELDS
        or payload.get("gates")
        != {
            "minimum_named_stock_count": 3,
            "minimum_rows_per_stock": 16,
            "minimum_distinct_authors_per_stock": 5,
            "all_rows_require_explicit_derivative_rights": True,
            "all_rows_require_source_url": True,
            "all_rows_require_roll_process_scanner_lineage": True,
            "all_paths_unique": True,
            "all_manifest_sha256_unique": True,
            "pixel_file_reads": 0,
        }
    ):
        raise LocalVision3SourceGateError("BU5 frozen contract drift")
    _relative_path(str(payload.get("manifest", {}).get("path", "")))
    return payload


def _present(row: dict[str, Any], field: str) -> bool:
    return field in row and row[field] not in (None, "", [], {})


def audit_local_source(config: dict[str, Any], root: Path) -> dict[str, Any]:
    manifest_path = root / _relative_path(config["manifest"]["path"])
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != config["manifest"]["bytes"]
        or hash_file(manifest_path) != config["manifest"]["sha256"]
    ):
        raise LocalVision3SourceGateError("BU5 manifest integrity mismatch")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocalVision3SourceGateError(
                    f"BU5 invalid manifest JSON at line {line_number}"
                ) from exc
            if row.get("style") in STYLES:
                rows.append(row)
    counts = {style: 0 for style in STYLES}
    authors: dict[str, set[str]] = {style: set() for style in STYLES}
    field_counts = {
        style: {field: 0 for field in REQUIRED_FIELDS} for style in STYLES
    }
    paths: list[str] = []
    hashes: list[str] = []
    explicit_rights_count = 0
    lineage_count = 0
    source_url_count = 0
    for row in rows:
        style = str(row["style"])
        counts[style] += 1
        for field in REQUIRED_FIELDS:
            field_counts[style][field] += int(_present(row, field))
        if _present(row, "author"):
            authors[style].add(str(row["author"]))
        if _present(row, "path"):
            paths.append(str(row["path"]))
        if _present(row, "sha256"):
            hashes.append(str(row["sha256"]))
        explicit_rights_count += int(
            row.get("redistributable") is True
            and _present(row, "license")
            and not str(row["license"]).lower().startswith("unknown")
        )
        source_url_count += int(_present(row, "source_url"))
        lineage_count += int(
            all(
                _present(row, field)
                for field in ("roll_id", "process_id", "scanner_id")
            )
        )
    author_counts = {style: len(authors[style]) for style in STYLES}
    populated_stocks = sum(count > 0 for count in counts.values())
    gate_results = {
        "minimum_named_stock_count": populated_stocks
        >= int(config["gates"]["minimum_named_stock_count"]),
        "minimum_rows_per_stock": all(
            counts[style] >= int(config["gates"]["minimum_rows_per_stock"])
            for style in STYLES
        ),
        "minimum_distinct_authors_per_stock": all(
            author_counts[style]
            >= int(config["gates"]["minimum_distinct_authors_per_stock"])
            for style in STYLES
        ),
        "all_rows_require_explicit_derivative_rights": explicit_rights_count
        == len(rows),
        "all_rows_require_source_url": source_url_count == len(rows),
        "all_rows_require_roll_process_scanner_lineage": lineage_count == len(rows),
        "all_paths_unique": len(paths) == len(rows) == len(set(paths)),
        "all_manifest_sha256_unique": len(hashes) == len(rows) == len(set(hashes)),
        "pixel_file_reads": True,
    }
    gate_pass = all(gate_results.values())
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "manifest_sha256": config["manifest"]["sha256"],
        "selected_row_count": len(rows),
        "row_counts_by_style": counts,
        "distinct_author_counts_by_style": author_counts,
        "required_field_counts_by_style": field_counts,
        "explicit_rights_row_count": explicit_rights_count,
        "source_url_row_count": source_url_count,
        "roll_process_scanner_lineage_row_count": lineage_count,
        "pixel_file_reads": 0,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable_payload)).hexdigest(),
        "gate_pass": gate_pass,
        "decision": (
            "open_integrity_and_nuisance_audit"
            if gate_pass
            else "quarantine_local_vision3_corpus_from_source_signature_validation"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
