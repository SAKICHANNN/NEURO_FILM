"""Frozen-set and severe-artifact adjudication primitives for FilmCase.

The module only records reproducible input identity and human/vision review
outcomes. It never changes pixels or treats diagnostics as a substitute for a
reviewed severe-artifact verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EVALUATION_SCHEMA_VERSION = 1
SEVERITIES = frozenset({"none", "moderate", "severe", "uncertain"})
REQUIRED_GOLD_BUCKETS = frozenset({"face", "text_logo", "sky", "foliage", "high_key", "deep_shadow", "saturated_objects", "fine_detail"})


class EvaluationContractError(ValueError):
    """Raised when a frozen-set or adjudication contract is invalid."""


@dataclass(frozen=True)
class EvaluationResult:
    """Validated output of an artifact adjudication run."""

    report: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"cannot read {path}: {exc}") from exc


def freeze_union_sources(
    source_path: Path,
    *,
    root: Path,
    gold_ids: Iterable[str],
    bucket_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Freeze source identities from the ignored union source index.

    A missing local image is recorded as unavailable rather than silently
    omitted. The resulting set remains provisional until required category
    coverage and reviewed adjudications satisfy the U4 gate.
    """

    raw = _read_json(source_path)
    if not isinstance(raw, list):
        raise EvaluationContractError("union source index must contain a list")
    wanted = {str(identifier).zfill(2) for identifier in gold_ids}
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("input"), str):
            raise EvaluationContractError("union source row requires string id and input")
        sample_id = item["id"].zfill(2)
        if sample_id in seen:
            raise EvaluationContractError(f"duplicate union source id: {sample_id}")
        seen.add(sample_id)
        source = Path(item["input"])
        exists = source.is_file()
        samples.append(
            {
                "id": sample_id,
                "key": item.get("key"),
                "source_path": str(source.relative_to(root)).replace("\\", "/") if exists and source.is_relative_to(root) else str(source),
                "source_sha256": sha256_file(source) if exists else None,
                "availability": "available" if exists else "missing",
                "split": "gold" if sample_id in wanted else "stress",
                "buckets": sorted(set(bucket_map.get(sample_id, []))),
            }
        )
    unknown = wanted - seen
    if unknown:
        raise EvaluationContractError(f"gold ids absent from union source index: {sorted(unknown)}")
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "provisional",
        "source_index": str(source_path.relative_to(root)).replace("\\", "/") if source_path.is_relative_to(root) else str(source_path),
        "source_index_sha256": sha256_file(source_path),
        "required_gold_buckets": sorted(REQUIRED_GOLD_BUCKETS),
        "samples": samples,
        "limitations": [
            "This union-derived set is a reproducible local seed, not the final U4 severe-artifact gold set.",
            "Missing required category coverage and unreviewed samples block final-gold promotion.",
            "Metrics and hashes support replay only; severe-artifact outcomes require recorded visual adjudication."
        ],
    }


def coverage_report(frozen_set: dict[str, Any]) -> dict[str, Any]:
    samples = frozen_set.get("samples")
    if not isinstance(samples, list):
        raise EvaluationContractError("frozen set requires samples")
    gold = [row for row in samples if row.get("split") == "gold"]
    bucket_counts: Counter[str] = Counter(bucket for row in gold for bucket in row.get("buckets", []))
    available_gold = sum(row.get("availability") == "available" for row in gold)
    missing = sorted(REQUIRED_GOLD_BUCKETS - set(bucket_counts))
    return {
        "gold_count": len(gold),
        "available_gold_count": available_gold,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "missing_required_gold_buckets": missing,
        "coverage_complete": available_gold == len(gold) and not missing,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def adjudicate(frozen_set: dict[str, Any], records: Iterable[dict[str, Any]], *, candidate_id: str) -> EvaluationResult:
    """Aggregate one candidate's recorded severe-artifact decisions.

    Gold promotion fails closed on missing/uncertain reviews, unavailable inputs,
    or any recorded severe verdict. Stress reporting retains a Wilson interval.
    """

    rows = frozen_set.get("samples")
    if not isinstance(rows, list):
        raise EvaluationContractError("frozen set requires samples")
    by_id = {str(row.get("id")): row for row in rows}
    decisions: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("candidate_id") != candidate_id:
            continue
        sample_id = str(record.get("sample_id", "")).zfill(2)
        severity = record.get("severity")
        if sample_id not in by_id:
            raise EvaluationContractError(f"adjudication names unknown sample {sample_id}")
        if severity not in SEVERITIES:
            raise EvaluationContractError(f"invalid severity for {sample_id}: {severity!r}")
        if sample_id in decisions:
            raise EvaluationContractError(f"duplicate adjudication for {candidate_id}/{sample_id}")
        decisions[sample_id] = record
    split_reports: dict[str, dict[str, Any]] = {}
    for split in ("gold", "stress"):
        samples = [row for row in rows if row.get("split") == split]
        reviewed = [decisions[str(row["id"])] for row in samples if str(row["id"]) in decisions]
        severe = [item for item in reviewed if item["severity"] == "severe"]
        uncertain = [item for item in reviewed if item["severity"] == "uncertain"]
        unavailable = [row["id"] for row in samples if row.get("availability") != "available"]
        split_reports[split] = {
            "sample_count": len(samples),
            "reviewed_count": len(reviewed),
            "severe_count": len(severe),
            "uncertain_count": len(uncertain),
            "unreviewed_ids": [row["id"] for row in samples if str(row["id"]) not in decisions],
            "unavailable_ids": unavailable,
            "severe_rate": len(severe) / len(reviewed) if reviewed else None,
            "severe_rate_wilson95": _wilson_interval(len(severe), len(reviewed)),
        }
    gold = split_reports["gold"]
    coverage = coverage_report(frozen_set)
    gold_pass = (
        frozen_set.get("status") == "final"
        and coverage["coverage_complete"]
        and gold["reviewed_count"] == gold["sample_count"]
        and not gold["unavailable_ids"]
        and gold["severe_count"] == 0
        and gold["uncertain_count"] == 0
    )
    return EvaluationResult({"schema_version": EVALUATION_SCHEMA_VERSION, "candidate_id": candidate_id, "coverage": coverage, "splits": split_reports, "gold_promotion_pass": gold_pass})
