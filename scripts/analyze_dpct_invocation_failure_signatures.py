#!/usr/bin/env python3
"""Aggregate deterministic failure signatures from a frozen P44 progress file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping


INPUT_PROTOCOL = "neuro-film.dpct-invocation-promotion-evaluation.v1"
OUTPUT_SCHEMA = "neuro-film.dpct-invocation-failure-signatures.v1"
SAMPLE_IDS = ("01", "02", "03", "05", "07", "09")
EXPECTED_KNOWN_IDS = tuple(
    f"{source}->{reference}"
    for source in SAMPLE_IDS
    for reference in SAMPLE_IDS
    if source != reference
)
_INVOCATION_KEYS = {
    "clipping_fraction",
    "consumer_receipt_id",
    "consumer_transform_id",
    "out_of_gamut_fraction",
    "producer_bundle_id",
    "producer_diagnostics_id",
    "producer_result_id",
    "projected_fraction",
    "request_id",
    "response_id",
}
_INVOCATION_OPTIONAL_KEYS = {
    # P44 added the exact output-pixel identity when the generic invocation
    # adapter replaced the original D-PCT-only adapter.  It is validated here
    # but deliberately excluded from the timing-independent failure signature.
    "output_pixel_sha256",
}
_KNOWN_METRIC_KEYS = {
    "candidate_new_boundary_fraction",
    "candidate_target_delta_e76_median",
    "candidate_target_delta_e76_p95",
    "median_improvement_fraction",
    "sample_id",
    "source_target_delta_e76_median",
    "source_target_delta_e76_p95",
}
_PHOTO_METRIC_KEYS = {
    "neutral_chroma_p95",
    "new_boundary_fraction",
    "passed",
    "reasons",
    "semantic_hue_rotation_p95_degrees",
    "tone_largest_reversal_delta_l",
    "tone_plateau_fraction",
    "tone_reversal_fraction",
}
_CONTEXT_METRIC_KEYS = {
    "delta_e76_maximum",
    "delta_e76_median",
    "delta_e76_p95",
    "passed",
    "reasons",
}


class FailureSignatureError(ValueError):
    """Raised when P44 progress is incomplete or internally inconsistent."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FailureSignatureError("analysis is not canonicalizable") from exc


def _strict(
    value: Any,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise FailureSignatureError(f"{label} fields are invalid")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FailureSignatureError(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise FailureSignatureError(f"{label} is below its minimum")
    return number


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise FailureSignatureError(f"{label} is invalid")
    return value


def _invocation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureSignatureError(f"{label} fields are invalid")
    keys = set(value)
    if keys not in (
        _INVOCATION_KEYS,
        _INVOCATION_KEYS | _INVOCATION_OPTIONAL_KEYS,
    ):
        raise FailureSignatureError(f"{label} fields are invalid")
    item = dict(value)
    for key in (
        "consumer_receipt_id",
        "consumer_transform_id",
        "producer_bundle_id",
        "producer_diagnostics_id",
        "producer_result_id",
        "request_id",
        "response_id",
    ):
        _identifier(item[key], f"{label}.{key}")
    if "output_pixel_sha256" in item:
        output_sha = item["output_pixel_sha256"]
        if (
            not isinstance(output_sha, str)
            or len(output_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in output_sha
            )
        ):
            raise FailureSignatureError(
                f"{label}.output_pixel_sha256 is invalid"
            )
    for key in (
        "clipping_fraction",
        "out_of_gamut_fraction",
        "projected_fraction",
    ):
        number = _finite(item[key], f"{label}.{key}", minimum=0.0)
        if number > 1.0:
            raise FailureSignatureError(f"{label}.{key} exceeds one")
        item[key] = number
    return item


def _median(values: list[float]) -> float:
    if not values:
        raise FailureSignatureError("cannot aggregate an empty series")
    return float(statistics.median(values))


def _pearson(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        raise FailureSignatureError("correlation series are invalid")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = sum(
        (x - first_mean) * (y - second_mean)
        for x, y in zip(first, second, strict=True)
    )
    first_energy = sum((x - first_mean) ** 2 for x in first)
    second_energy = sum((y - second_mean) ** 2 for y in second)
    if first_energy == 0.0 or second_energy == 0.0:
        return 0.0
    return float(numerator / math.sqrt(first_energy * second_energy))


def _validate_progress(value: Any) -> Mapping[str, Any]:
    progress = _strict(
        value,
        {
            "protocol",
            "contract_id",
            "known_rows",
            "photographic_rows",
            "context_rows",
        },
        "progress",
    )
    if progress["protocol"] != INPUT_PROTOCOL:
        raise FailureSignatureError("progress protocol is unsupported")
    _identifier(progress["contract_id"], "contract_id")
    known = progress["known_rows"]
    photo = progress["photographic_rows"]
    context = progress["context_rows"]
    if (
        not isinstance(known, Mapping)
        or tuple(sorted(known)) != tuple(sorted(EXPECTED_KNOWN_IDS))
        or not isinstance(photo, Mapping)
        or tuple(sorted(photo)) != tuple(sorted(SAMPLE_IDS))
        or not isinstance(context, Mapping)
        or tuple(sorted(context)) != tuple(sorted(SAMPLE_IDS))
    ):
        raise FailureSignatureError("P44 row inventory is incomplete")
    return progress


def _known_rows(progress: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id in EXPECTED_KNOWN_IDS:
        row = _strict(
            progress["known_rows"][row_id],
            {"invocation", "metrics"},
            f"known_rows.{row_id}",
        )
        invocation = _invocation(
            row["invocation"], f"known_rows.{row_id}.invocation"
        )
        metrics = dict(
            _strict(
                row["metrics"],
                _KNOWN_METRIC_KEYS,
                f"known_rows.{row_id}.metrics",
            )
        )
        if metrics["sample_id"] != row_id:
            raise FailureSignatureError("known sample identity mismatch")
        numbers = {
            key: _finite(
                metrics[key],
                f"known_rows.{row_id}.{key}",
                minimum=(0.0 if key != "median_improvement_fraction" else None),
            )
            for key in _KNOWN_METRIC_KEYS
            if key != "sample_id"
        }
        source_error = numbers["source_target_delta_e76_median"]
        candidate_error = numbers["candidate_target_delta_e76_median"]
        if source_error <= 0.0:
            raise FailureSignatureError("source error must be positive")
        expected_improvement = (source_error - candidate_error) / source_error
        if not math.isclose(
            numbers["median_improvement_fraction"],
            expected_improvement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FailureSignatureError("known improvement is inconsistent")
        source_id, reference_id = row_id.split("->")
        rows.append(
            {
                "row_id": row_id,
                "source_id": source_id,
                "reference_id": reference_id,
                "source_error": source_error,
                "candidate_error": candidate_error,
                "candidate_source_error_ratio": candidate_error / source_error,
                "improvement": numbers["median_improvement_fraction"],
                "new_boundary": numbers["candidate_new_boundary_fraction"],
                "clipping": invocation["clipping_fraction"],
                "bundle_id": invocation["producer_bundle_id"],
                "transform_id": invocation["consumer_transform_id"],
            }
        )
    return rows


def _strata(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sample_id in SAMPLE_IDS:
        selected = [row for row in rows if row[field] == sample_id]
        result[sample_id] = {
            "row_count": len(selected),
            "median_improvement_fraction": _median(
                [row["improvement"] for row in selected]
            ),
            "worst_improvement_fraction": min(
                row["improvement"] for row in selected
            ),
            "median_candidate_source_error_ratio": _median(
                [row["candidate_source_error_ratio"] for row in selected]
            ),
            "maximum_new_boundary_fraction": max(
                row["new_boundary"] for row in selected
            ),
            "median_clipping_fraction": _median(
                [row["clipping"] for row in selected]
            ),
            "unique_bundle_count": len(
                {row["bundle_id"] for row in selected}
            ),
        }
    return result


def _photo(progress: Mapping[str, Any]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    metrics_rows: list[Mapping[str, Any]] = []
    clipping: list[float] = []
    bundle_ids: list[str] = []
    for sample_id in SAMPLE_IDS:
        row = _strict(
            progress["photographic_rows"][sample_id],
            {"invocation", "metrics"},
            f"photographic_rows.{sample_id}",
        )
        invocation = _invocation(
            row["invocation"],
            f"photographic_rows.{sample_id}.invocation",
        )
        metrics = _strict(
            row["metrics"],
            _PHOTO_METRIC_KEYS,
            f"photographic_rows.{sample_id}.metrics",
        )
        if not isinstance(metrics["passed"], bool):
            raise FailureSignatureError("photographic pass state is invalid")
        if not isinstance(metrics["reasons"], list) or any(
            not isinstance(item, str) or not item
            for item in metrics["reasons"]
        ):
            raise FailureSignatureError("photographic reasons are invalid")
        if metrics["passed"] != (len(metrics["reasons"]) == 0):
            raise FailureSignatureError(
                "photographic pass/reasons are inconsistent"
            )
        for key in _PHOTO_METRIC_KEYS - {"passed", "reasons"}:
            _finite(metrics[key], f"photographic_rows.{sample_id}.{key}")
        for reason in metrics["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        metrics_rows.append(metrics)
        clipping.append(invocation["clipping_fraction"])
        bundle_ids.append(invocation["producer_bundle_id"])
    return {
        "sample_count": len(metrics_rows),
        "failed_count": sum(not row["passed"] for row in metrics_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "maximum_neutral_chroma_p95": max(
            float(row["neutral_chroma_p95"]) for row in metrics_rows
        ),
        "maximum_new_boundary_fraction": max(
            float(row["new_boundary_fraction"]) for row in metrics_rows
        ),
        "maximum_semantic_hue_rotation_p95_degrees": max(
            float(row["semantic_hue_rotation_p95_degrees"])
            for row in metrics_rows
        ),
        "median_clipping_fraction": _median(clipping),
        "unique_bundle_count": len(set(bundle_ids)),
    }


def _context(progress: Mapping[str, Any]) -> dict[str, Any]:
    metrics_rows: list[Mapping[str, Any]] = []
    changed_bundle_pairs = 0
    clipping: list[float] = []
    for sample_id in SAMPLE_IDS:
        row = _strict(
            progress["context_rows"][sample_id],
            {"invocations", "metrics"},
            f"context_rows.{sample_id}",
        )
        if not isinstance(row["invocations"], list) or len(row["invocations"]) != 2:
            raise FailureSignatureError(
                "context row must contain exactly two invocations"
            )
        invocations = [
            _invocation(
                item,
                f"context_rows.{sample_id}.invocations[{index}]",
            )
            for index, item in enumerate(row["invocations"])
        ]
        changed_bundle_pairs += int(
            invocations[0]["producer_bundle_id"]
            != invocations[1]["producer_bundle_id"]
        )
        clipping.extend(item["clipping_fraction"] for item in invocations)
        metrics = _strict(
            row["metrics"],
            _CONTEXT_METRIC_KEYS,
            f"context_rows.{sample_id}.metrics",
        )
        if not isinstance(metrics["passed"], bool):
            raise FailureSignatureError("context pass state is invalid")
        if not isinstance(metrics["reasons"], list) or any(
            not isinstance(item, str) or not item
            for item in metrics["reasons"]
        ):
            raise FailureSignatureError("context reasons are invalid")
        if metrics["passed"] != (len(metrics["reasons"]) == 0):
            raise FailureSignatureError("context pass/reasons are inconsistent")
        for key in _CONTEXT_METRIC_KEYS - {"passed", "reasons"}:
            _finite(metrics[key], f"context_rows.{sample_id}.{key}", minimum=0)
        metrics_rows.append(metrics)
    return {
        "sample_count": len(metrics_rows),
        "failed_count": sum(not row["passed"] for row in metrics_rows),
        "changed_bundle_pair_count": changed_bundle_pairs,
        "maximum_delta_e76_median": max(
            float(row["delta_e76_median"]) for row in metrics_rows
        ),
        "maximum_delta_e76_p95": max(
            float(row["delta_e76_p95"]) for row in metrics_rows
        ),
        "maximum_delta_e76": max(
            float(row["delta_e76_maximum"]) for row in metrics_rows
        ),
        "median_invocation_clipping_fraction": _median(clipping),
    }


def analyze_progress(value: Any) -> dict[str, Any]:
    """Return a canonical, timing-independent P44 diagnostic."""

    progress = _validate_progress(value)
    rows = _known_rows(progress)
    ratios = [row["candidate_source_error_ratio"] for row in rows]
    improvements = [row["improvement"] for row in rows]
    clipping = [row["clipping"] for row in rows]
    boundaries = [row["new_boundary"] for row in rows]
    photo = _photo(progress)
    context = _context(progress)
    regressed = [row for row in rows if row["improvement"] < 0.0]
    low_clip_regressed = [
        row for row in regressed if row["clipping"] <= 0.05
    ]
    known = {
        "row_count": len(rows),
        "regressed_count": len(regressed),
        "candidate_error_exceeds_source_count": sum(
            ratio > 1.0 for ratio in ratios
        ),
        "median_candidate_source_error_ratio": _median(ratios),
        "minimum_candidate_source_error_ratio": min(ratios),
        "maximum_candidate_source_error_ratio": max(ratios),
        "median_improvement_fraction": _median(improvements),
        "worst_improvement_fraction": min(improvements),
        "maximum_new_boundary_fraction": max(boundaries),
        "median_clipping_fraction": _median(clipping),
        "regressed_with_clipping_at_most_5pct_count": len(
            low_clip_regressed
        ),
        "clipping_error_ratio_pearson": _pearson(clipping, ratios),
        "unique_bundle_count": len({row["bundle_id"] for row in rows}),
        "unique_transform_count": len(
            {row["transform_id"] for row in rows}
        ),
        "by_source": _strata(rows, "source_id"),
        "by_reference": _strata(rows, "reference_id"),
    }
    signatures = {
        "universal_global_overcorrection": (
            known["candidate_error_exceeds_source_count"]
            == known["row_count"]
        ),
        "regression_predominantly_not_explained_by_large_clipping": (
            known["regressed_with_clipping_at_most_5pct_count"]
            >= math.ceil(0.9 * known["regressed_count"])
        ),
        "source_context_changes_every_context_bundle": (
            context["changed_bundle_pair_count"] == context["sample_count"]
        ),
        "neutral_axis_corruption_present": (
            photo["reason_counts"].get("neutral-axis-chroma", 0) > 0
        ),
        "semantic_hue_corruption_present": (
            photo["reason_counts"].get("semantic-hue-rotation", 0) > 0
        ),
        "boundary_corruption_present": (
            photo["reason_counts"].get("new-boundary-fraction", 0) > 0
        ),
        "all_context_probes_fail": (
            context["failed_count"] == context["sample_count"]
        ),
    }
    report: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "input_protocol": INPUT_PROTOCOL,
        "contract_id": progress["contract_id"],
        "known_operator": known,
        "photographic": photo,
        "context": context,
        "failure_signatures": signatures,
    }
    report["stable_evidence_id"] = hashlib.sha256(
        b"NeuroFilmDpctInvocationFailureSignaturesV1\0"
        + _canonical_json(report)
    ).hexdigest()
    return report


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    try:
        value = json.loads(args.input.read_text(encoding="utf-8"))
        report = analyze_progress(value)
        _atomic_write(args.output, report)
    except (OSError, json.JSONDecodeError, FailureSignatureError) as exc:
        raise SystemExit(f"failure-signature analysis failed closed: {exc}")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
