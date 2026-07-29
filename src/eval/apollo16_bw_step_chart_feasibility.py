"""P2L1 source-geometry audit for the AS16-111 B&W wedge segment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.apollo16_bw_step_chart_source import validate_config as validate_source
from src.eval.apollo_step_chart_source import sha256_file
from src.eval.classic_tiff_stream import summarize_classic_tiff_zip_member


SCHEMA = "neuro_film.u6_p2l1_apollo16_bw_step_chart_feasibility.v1"
REPORT_SCHEMA = "neuro_film.u6_p2l1_apollo16_bw_step_chart_feasibility_report.v1"


class Apollo16BWWedgeFeasibilityError(ValueError):
    """Raised when the P2L1 evidence or analysis contract drifts."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256(contiguous.tobytes(order="C"))


def _read_exact_json(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    raw = path.read_bytes()
    if _sha256(raw) != str(binding["sha256"]):
        raise Apollo16BWWedgeFeasibilityError(f"hash drift: {binding['path']}")
    return json.loads(raw)


def validate_contract(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise Apollo16BWWedgeFeasibilityError("unsupported P2L1 contract")
    source = _read_exact_json(root, config["parents"]["source_contract"])
    validate_source(source)
    manifest_binding = config["parents"]["acquisition_manifest"]
    manifest = _read_exact_json(root, manifest_binding)
    if (
        manifest.get("sha256") != manifest_binding["archive_sha256"]
        or manifest.get("compressed_bytes") != manifest_binding["archive_bytes"]
        or manifest.get("member_count") != 1
        or manifest.get("zip_crc_pass") is not True
    ):
        raise Apollo16BWWedgeFeasibilityError("acquisition identity drifted")
    member = manifest["members"][0]
    if (
        member.get("path") != manifest_binding["member_path"]
        or member.get("uncompressed_bytes")
        != manifest_binding["member_uncompressed_bytes"]
        or member.get("crc32") != manifest_binding["member_crc32"]
    ):
        raise Apollo16BWWedgeFeasibilityError("member identity drifted")
    archive = root / str(source["acquisition"]["destination"])
    if (
        not archive.is_file()
        or archive.stat().st_size != manifest_binding["archive_bytes"]
        or sha256_file(archive) != manifest_binding["archive_sha256"]
    ):
        raise Apollo16BWWedgeFeasibilityError("archive bytes drifted")
    disclosure = config.get("development_disclosure", {})
    if not all(
        disclosure.get(key) is True
        for key in (
            "sampled_preview_seen_before_numeric_gate_freeze",
            "two_endpoint_like_regions_seen",
            "untouched_confirmation_claim_forbidden",
        )
    ):
        raise Apollo16BWWedgeFeasibilityError("development disclosure drifted")
    return source


def supported_level_groups(
    profile_u16: np.ndarray,
    *,
    bin_width_u16: int,
    minimum_support_fraction: float,
) -> list[dict[str, Any]]:
    """Return adjacent groups of histogram bins with material row support."""

    profile = np.asarray(profile_u16, dtype=np.float64)
    if (
        profile.ndim != 1
        or len(profile) == 0
        or not np.isfinite(profile).all()
        or bin_width_u16 <= 0
        or not 0.0 < minimum_support_fraction <= 1.0
        or np.any((profile < 0.0) | (profile > 65535.0))
    ):
        raise Apollo16BWWedgeFeasibilityError("invalid level profile")
    bins = np.floor(profile / float(bin_width_u16)).astype(np.int64)
    identities, counts = np.unique(bins, return_counts=True)
    minimum_count = int(math.ceil(minimum_support_fraction * len(profile)))
    supported = [
        int(identity)
        for identity, count in zip(identities, counts, strict=True)
        if int(count) >= minimum_count
    ]
    groups: list[list[int]] = []
    for identity in supported:
        if not groups or identity > groups[-1][-1] + 1:
            groups.append([identity])
        else:
            groups[-1].append(identity)
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = np.isin(bins, np.asarray(group))
        output.append(
            {
                "first_bin": group[0],
                "last_bin": group[-1],
                "support_rows": int(np.count_nonzero(selected)),
                "support_fraction": float(np.mean(selected)),
                "mean_code": float(np.mean(profile[selected])),
                "minimum_code": float(np.min(profile[selected])),
                "maximum_code": float(np.max(profile[selected])),
            }
        )
    return output


def evaluate_apollo16_bw_step_chart_feasibility(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    source = validate_contract(config, root)
    analysis = config["analysis"]
    gates = config["automatic_gates"]
    manifest_binding = config["parents"]["acquisition_manifest"]
    summary = summarize_classic_tiff_zip_member(
        root / str(source["acquisition"]["destination"]),
        str(manifest_binding["member_path"]),
        maximum_sample_rows=int(analysis["maximum_sample_rows"]),
        maximum_sample_columns=int(analysis["maximum_sample_columns"]),
        maximum_uncompressed_bytes=int(analysis["maximum_uncompressed_bytes"]),
    )
    sampled = summary.sampled_u16
    left = int(
        np.floor(
            float(analysis["central_column_fraction"][0]) * sampled.shape[1]
        )
    )
    right = int(
        np.ceil(
            float(analysis["central_column_fraction"][1]) * sampled.shape[1]
        )
    )
    if left < 0 or right > sampled.shape[1] or right <= left:
        raise Apollo16BWWedgeFeasibilityError("central crop is invalid")
    profile = np.median(sampled[:, left:right, 0], axis=1)
    groups = supported_level_groups(
        profile,
        bin_width_u16=int(analysis["level_bin_width_u16"]),
        minimum_support_fraction=float(
            analysis["minimum_level_support_fraction"]
        ),
    )
    code_span_fraction = float(
        (np.max(profile) - np.min(profile)) / 65535.0
    )
    layout = summary.layout
    checks = {
        "width": layout.width == int(gates["required_width"]),
        "height": layout.height == int(gates["required_height"]),
        "samples_per_pixel": layout.samples_per_pixel
        == int(gates["required_samples_per_pixel"]),
        "bits_per_sample": layout.bits_per_sample
        == (int(gates["required_bits_per_sample"]),),
        "rows_per_strip": layout.rows_per_strip
        == int(gates["required_rows_per_strip"]),
        "code_span": code_span_fraction
        >= float(gates["minimum_code_span_fraction"]),
        "supported_wedge_levels": len(groups)
        >= int(gates["minimum_supported_wedge_levels"]),
    }
    automatic_pass = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "source": {
            "archive_sha256": str(manifest_binding["archive_sha256"]),
            "archive_bytes": int(manifest_binding["archive_bytes"]),
            "member_path": str(manifest_binding["member_path"]),
            "member_crc32": str(manifest_binding["member_crc32"]),
        },
        "layout": {
            "width": layout.width,
            "height": layout.height,
            "samples_per_pixel": layout.samples_per_pixel,
            "bits_per_sample": list(layout.bits_per_sample),
            "rows_per_strip": layout.rows_per_strip,
            "strip_count": len(layout.strip_offsets),
            "photometric": layout.photometric,
        },
        "analysis": {
            "sampled_shape": list(sampled.shape),
            "sampled_u16_sha256": _array_sha256(sampled),
            "row_means_sha256": _array_sha256(summary.row_means),
            "column_means_sha256": _array_sha256(summary.column_means),
            "profile_sha256": _array_sha256(profile),
            "profile_minimum_code": float(np.min(profile)),
            "profile_maximum_code": float(np.max(profile)),
            "code_span_fraction": code_span_fraction,
            "supported_level_count": len(groups),
            "supported_level_groups": groups,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_bw_response_source_feasibility_only"
            if automatic_pass
            else "close_segment_without_curve_fitting_insufficient_wedge_levels"
        ),
        "development_disclosure": dict(config["development_disclosure"]),
        "claim_ceiling": str(config["claim_ceiling"]),
    }
    report["stable_evidence_id"] = _sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return report


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "Apollo16BWWedgeFeasibilityError",
    "evaluate_apollo16_bw_step_chart_feasibility",
    "supported_level_groups",
    "validate_contract",
    "write_report",
]
