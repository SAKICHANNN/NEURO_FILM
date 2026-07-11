"""Lineage-first manifest auditing for the autonomous FilmCase research lane.

The legacy training manifest is intentionally left untouched. FilmCase needs a
stricter contract: a reference image is usable only when its source grouping is
known well enough to make group-based splitting meaningful. Rows without that
evidence are retained in the audit output but quarantined from case memory,
router supervision, and identifiability claims.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


MANIFEST_SCHEMA_VERSION = 2
_IMAGE_HASH_SIZE = (9, 8)
_DEFAULT_PHASH_THRESHOLD = 4
_REPORT_ISSUE_SAMPLE_LIMIT = 20
_SOURCE_URL_KEYS = ("source_url", "photo_url", "url")
_SOURCE_ID_KEYS = ("source_id", "photo_id", "id")
_UPLOADER_KEYS = ("uploader_id", "owner_id", "owner", "user_id")
_SCAN_GROUP_KEYS = ("scan_group", "roll_id", "capture_id", "lab_scan_id")


class ManifestAuditError(ValueError):
    """Raised when the audit input is structurally invalid."""


@dataclass(frozen=True)
class AuditIssue:
    """A concrete audit finding retained in the JSON report."""

    kind: str
    paths: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "paths": list(self.paths),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AuditResult:
    """Immutable audit result suitable for report and manifest serialization."""

    rows: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestAuditError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ManifestAuditError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ManifestAuditError(f"{path}: no manifest rows")
    return rows


def _metadata_by_filename(root: Path, manifest_rows: Iterable[dict[str, Any]]) -> dict[Path, dict[str, dict[str, Any]]]:
    """Load optional sidecars once per image directory without inferring provenance."""

    directories = {
        (root / str(row["path"])).parent
        for row in manifest_rows
        if isinstance(row.get("path"), str)
    }
    result: dict[Path, dict[str, dict[str, Any]]] = {}
    for directory in directories:
        sidecar = directory / "metadata.jsonl"
        if not sidecar.exists():
            continue
        indexed: dict[str, dict[str, Any]] = {}
        for record in _read_jsonl(sidecar):
            file_name = record.get("file_name")
            if isinstance(file_name, str) and file_name:
                indexed[file_name] = record
        result[directory] = indexed
    return result


def _dhash(path: Path) -> int:
    """Return a deterministic 64-bit difference hash for a local raster."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                pixels = image.convert("L").resize(_IMAGE_HASH_SIZE, Image.Resampling.LANCZOS)
                values = list(pixels.get_flattened_data())
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ManifestAuditError(f"{path}: cannot compute perceptual hash: {type(exc).__name__}") from exc

    digest = 0
    width, height = _IMAGE_HASH_SIZE
    for y in range(height):
        row_start = y * width
        for x in range(width - 1):
            digest = (digest << 1) | int(values[row_start + x] > values[row_start + x + 1])
    return digest


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _near_duplicate_pairs(
    hashes: dict[str, int],
    threshold: int,
) -> list[tuple[str, str, int]]:
    """Find dHash-near pairs without an all-pairs scan.

    A pair within distance four shares at least one of the four 16-bit chunks,
    so chunk buckets provide a complete candidate set for the default threshold.
    """

    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for path, value in hashes.items():
        for chunk_index in range(4):
            chunk = (value >> (16 * chunk_index)) & 0xFFFF
            buckets[(chunk_index, chunk)].append(path)

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, int]] = []
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                distance = _hamming_distance(hashes[left], hashes[right])
                if distance <= threshold:
                    pairs.append((key[0], key[1], distance))
    return sorted(pairs)


def _lineage_fields(
    row: dict[str, Any],
    sidecar: dict[str, Any] | None,
) -> tuple[dict[str, str | None], str, str | None]:
    combined = dict(row)
    if sidecar:
        combined.update({key: value for key, value in sidecar.items() if value is not None})

    source_url = _first_nonempty(combined, _SOURCE_URL_KEYS)
    source_id = _first_nonempty(combined, _SOURCE_ID_KEYS)
    uploader_id = _first_nonempty(combined, _UPLOADER_KEYS)
    scan_group = _first_nonempty(combined, _SCAN_GROUP_KEYS)
    caption_only = bool(sidecar and sidecar.get("text"))

    if (source_url or source_id) and (uploader_id or scan_group):
        status = "resolved_group"
        group_id = f"scan:{scan_group}" if scan_group else f"uploader:{uploader_id}"
    elif source_url or source_id:
        status = "partial_sample_only"
        group_id = None
    elif caption_only:
        status = "caption_only"
        group_id = None
    else:
        status = "unresolved"
        group_id = None

    return (
        {
            "source_url": source_url,
            "source_id": source_id,
            "uploader_id": uploader_id,
            "scan_group": scan_group,
        },
        status,
        group_id,
    )


def _group_split(group_id: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def audit_manifest(
    manifest_path: Path,
    *,
    root: Path,
    validation_percent: int = 10,
    perceptual_threshold: int = _DEFAULT_PHASH_THRESHOLD,
    compute_perceptual_hashes: bool = True,
) -> AuditResult:
    """Audit a legacy manifest and return a quarantined FilmCase manifest v2.

    The function never writes the source manifest and never upgrades uncertain
    provenance by guesswork. Missing group lineage produces `quarantine`.
    """

    if not 0 <= validation_percent < 100:
        raise ManifestAuditError("validation_percent must be in [0, 99]")
    if perceptual_threshold < 0 or perceptual_threshold > 4:
        raise ManifestAuditError("perceptual_threshold must be in [0, 4]")

    manifest_path = manifest_path.resolve()
    root = root.resolve()
    legacy_rows = _read_jsonl(manifest_path)
    for index, row in enumerate(legacy_rows, start=1):
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise ManifestAuditError(f"{manifest_path}: row {index} is missing path")

    metadata = _metadata_by_filename(root, legacy_rows)
    v2_rows: list[dict[str, Any]] = []
    issues: list[AuditIssue] = []
    strict_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    perceptual_hashes: dict[str, int] = {}
    metadata_rows = 0

    for legacy_row in legacy_rows:
        relative_path = Path(str(legacy_row["path"]).replace("\\", "/"))
        absolute_path = root / relative_path
        if not absolute_path.exists():
            issues.append(AuditIssue("missing_file", (str(relative_path),), "manifest path does not exist"))
            continue

        sidecar = metadata.get(absolute_path.parent, {}).get(absolute_path.name)
        if sidecar:
            metadata_rows += 1
        lineage, lineage_status, group_id = _lineage_fields(legacy_row, sidecar)
        source_license = str(legacy_row.get("license", "unknown")).strip() or "unknown"
        source = str(legacy_row.get("source", "unknown")).strip() or "unknown"
        sha256 = str(legacy_row.get("sha256", "")).strip() or None
        if not sha256:
            issues.append(AuditIssue("missing_sha256", (str(relative_path),), "legacy row lacks sha256"))

        split = _group_split(group_id, validation_percent) if group_id else "quarantine"
        eligibility = (
            "eligible_research_only"
            if group_id and source_license != "unknown-flickr-user-content"
            else "quarantined_missing_group_lineage"
        )
        if lineage_status != "resolved_group":
            issues.append(AuditIssue("lineage_quarantine", (str(relative_path),), lineage_status))

        row: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "path": str(relative_path).replace("\\", "/"),
            "legacy_split": legacy_row.get("split"),
            "filmcase_split": split,
            "source": source,
            "source_license": source_license,
            "allowed_use": "research-only",
            "redistributable": False,
            "style": legacy_row.get("style"),
            "task": legacy_row.get("task"),
            "sha256": sha256,
            "sha256_source": "legacy_manifest" if sha256 else "missing",
            "width": legacy_row.get("width"),
            "height": legacy_row.get("height"),
            "bytes": legacy_row.get("bytes"),
            "lineage_status": lineage_status,
            "source_group_id": group_id,
            "metadata_status": "caption_sidecar" if sidecar else "missing",
            "filmcase_eligibility": eligibility,
            **lineage,
        }
        if compute_perceptual_hashes:
            try:
                perceptual = _dhash(absolute_path)
            except ManifestAuditError as exc:
                issues.append(AuditIssue("perceptual_hash_failed", (str(relative_path),), str(exc)))
                perceptual = None
            row["dhash64"] = f"{perceptual:016x}" if perceptual is not None else None
            if perceptual is not None:
                perceptual_hashes[row["path"]] = perceptual

        v2_rows.append(row)
        if sha256:
            strict_hashes[sha256].append(row)

    exact_duplicates = [
        sorted(rows, key=lambda item: str(item["path"]))
        for rows in strict_hashes.values()
        if len(rows) > 1
    ]
    for group in exact_duplicates:
        issues.append(
            AuditIssue(
                "exact_duplicate",
                tuple(str(row["path"]) for row in group),
                "same sha256 appears in multiple manifest rows",
            )
        )

    near_pairs = _near_duplicate_pairs(perceptual_hashes, perceptual_threshold) if compute_perceptual_hashes else []
    for left, right, distance in near_pairs:
        issues.append(AuditIssue("perceptual_duplicate", (left, right), f"dhash distance={distance}"))

    cross_split_exact = sum(
        1
        for group in exact_duplicates
        if len({row["filmcase_split"] for row in group}) > 1
    )
    row_by_path = {str(row["path"]): row for row in v2_rows}
    cross_split_perceptual = sum(
        1
        for left, right, _ in near_pairs
        if row_by_path[left]["filmcase_split"] != row_by_path[right]["filmcase_split"]
    )
    status_counts: dict[str, int] = defaultdict(int)
    eligibility_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for row in v2_rows:
        status_counts[str(row["lineage_status"])] += 1
        eligibility_counts[str(row["filmcase_eligibility"])] += 1
        split_counts[str(row["filmcase_split"])] += 1

    has_resolved_groups = bool(status_counts["resolved_group"])
    gate_status = "ready_for_group_split" if has_resolved_groups else "blocked_missing_group_lineage"
    issue_counts: dict[str, int] = defaultdict(int)
    for issue in issues:
        issue_counts[issue.kind] += 1
    report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "manifest_rows_seen": len(legacy_rows),
        "manifest_rows_audited": len(v2_rows),
        "metadata_rows_matched": metadata_rows,
        "lineage_status_counts": dict(sorted(status_counts.items())),
        "filmcase_eligibility_counts": dict(sorted(eligibility_counts.items())),
        "filmcase_split_counts": dict(sorted(split_counts.items())),
        "duplicate_audit": {
            "exact_duplicate_groups": len(exact_duplicates),
            "perceptual_duplicate_pairs": len(near_pairs),
            "cross_filmcase_split_exact_groups": cross_split_exact,
            "cross_filmcase_split_perceptual_pairs": cross_split_perceptual,
            "perceptual_threshold": perceptual_threshold,
        },
        "gate_status": gate_status,
        "next_allowed_lane": "anchor_only" if not has_resolved_groups else "reference_identifiability",
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_sample_limit": _REPORT_ISSUE_SAMPLE_LIMIT,
        "issues": [issue.as_dict() for issue in issues[:_REPORT_ISSUE_SAMPLE_LIMIT]],
    }
    return AuditResult(rows=tuple(v2_rows), report=report)


def write_audit_outputs(
    result: AuditResult,
    *,
    report_path: Path,
    manifest_v2_path: Path,
) -> None:
    """Write new audit artifacts without modifying the legacy source manifest."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_v2_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_v2_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result.rows),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
