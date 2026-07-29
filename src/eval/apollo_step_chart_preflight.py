"""Bounded P2K layout and density-wedge geometry preflight.

This module deliberately stops before response-curve fitting.  It verifies the
exact acquired archive, streams the complete TIFF member, and measures only
layout, broad code span, and high-confidence horizontal separator geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from src.eval.apollo_step_chart_source import sha256_file, validate_config
from src.eval.classic_tiff_stream import summarize_classic_tiff_zip_member


SCHEMA = "neuro_film.u6_p2k_apollo_step_chart_preflight.v1"
REPORT_SCHEMA = "neuro_film.u6_p2k_apollo_step_chart_preflight_report.v1"


class ApolloStepChartPreflightError(ValueError):
    """Raised when exact P2K source or geometry evidence drifts."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(array).tobytes(order="C"))


def _read_bound_json(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    raw = path.read_bytes()
    if _sha256(raw) != str(binding["sha256"]):
        raise ApolloStepChartPreflightError(f"hash drift: {binding['path']}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ApolloStepChartPreflightError("bound JSON must be an object")
    return payload


def validate_contract(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config.get("status")
        != "post_exploratory_geometry_contract_frozen"
        or config.get("curve_fitting_allowed") is not False
        or config.get("operator_fitting_allowed") is not False
    ):
        raise ApolloStepChartPreflightError("unsupported P2K preflight contract")
    disclosure = config.get("development_disclosure", {})
    if not all(
        disclosure.get(key) is True
        for key in (
            "bounded_preview_seen_before_gate_freeze",
            "horizontal_step_geometry_seen",
            "untouched_confirmation_claim_forbidden",
        )
    ):
        raise ApolloStepChartPreflightError("development disclosure drifted")
    source = _read_bound_json(root, config["parents"]["source_contract"])
    validate_config(source)
    manifest = _read_bound_json(root, config["parents"]["acquisition_manifest"])
    expected = config["parents"]["acquisition_manifest"]
    members = manifest.get("members")
    if (
        manifest.get("schema")
        != "neuro_film.u6_p2k_apollo_step_chart_acquisition.v1"
        or manifest.get("sha256") != expected["archive_sha256"]
        or manifest.get("compressed_bytes") != expected["archive_bytes"]
        or manifest.get("zip_crc_pass") is not True
        or not isinstance(members, list)
        or len(members) != 1
        or members[0].get("path") != expected["member_path"]
        or members[0].get("uncompressed_bytes")
        != expected["member_uncompressed_bytes"]
        or members[0].get("crc32") != expected["member_crc32"]
    ):
        raise ApolloStepChartPreflightError("acquisition identity drifted")
    archive = root / str(source["acquisition"]["destination"])
    if (
        not archive.is_file()
        or archive.stat().st_size != expected["archive_bytes"]
        or sha256_file(archive) != expected["archive_sha256"]
    ):
        raise ApolloStepChartPreflightError("archive bytes drifted")
    return source, manifest


def high_confidence_negative_edges(
    profile: np.ndarray,
    *,
    smoothing_rows: int,
    minimum_drop_fraction: float,
    minimum_separation_rows: int,
) -> list[dict[str, float | int]]:
    """Locate separated downward code transitions without fitting a curve."""

    values = np.asarray(profile, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 65535.0)
        or smoothing_rows <= 0
        or smoothing_rows % 2 == 0
        or smoothing_rows > len(values)
        or not 0.0 < minimum_drop_fraction < 1.0
        or minimum_separation_rows <= 0
    ):
        raise ApolloStepChartPreflightError("invalid separator profile")
    kernel = np.full(smoothing_rows, 1.0 / smoothing_rows)
    smoothed = np.convolve(values / 65535.0, kernel, mode="same")
    differences = np.diff(smoothed)
    candidates = np.flatnonzero(differences <= -minimum_drop_fraction)
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: differences[item]):
        if all(abs(int(index) - prior) >= minimum_separation_rows for prior in selected):
            selected.append(int(index))
    return [
        {
            "sample_row_index": index,
            "normalized_drop": float(-differences[index]),
        }
        for index in sorted(selected)
    ]


def evaluate_apollo_step_chart_preflight(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], np.ndarray]:
    source, _ = validate_contract(config, root)
    analysis = config["analysis"]
    expected = config["parents"]["acquisition_manifest"]
    summary = summarize_classic_tiff_zip_member(
        root / str(source["acquisition"]["destination"]),
        str(expected["member_path"]),
        maximum_sample_rows=int(analysis["maximum_sample_rows"]),
        maximum_sample_columns=int(analysis["maximum_sample_columns"]),
        maximum_uncompressed_bytes=int(analysis["maximum_uncompressed_bytes"]),
    )
    sampled = summary.sampled_u16
    left = int(np.floor(float(analysis["central_column_fraction"][0]) * sampled.shape[1]))
    right = int(np.ceil(float(analysis["central_column_fraction"][1]) * sampled.shape[1]))
    if left < 0 or right > sampled.shape[1] or right <= left:
        raise ApolloStepChartPreflightError("central column crop is invalid")
    profile = np.median(sampled[:, left:right, :], axis=(1, 2))
    edges = high_confidence_negative_edges(
        profile,
        smoothing_rows=int(analysis["smoothing_rows"]),
        minimum_drop_fraction=float(analysis["minimum_separator_drop_fraction"]),
        minimum_separation_rows=int(analysis["minimum_separator_separation_rows"]),
    )
    top_count = max(1, int(np.ceil(len(profile) * float(analysis["endpoint_fraction"]))))
    top = float(np.median(profile[:top_count]))
    bottom = float(np.median(profile[-top_count:]))
    endpoint_span = float((bottom - top) / 65535.0)
    layout = summary.layout
    gates = config["automatic_gates"]
    checks = {
        "width": layout.width == int(gates["required_width"]),
        "height": layout.height == int(gates["required_height"]),
        "samples_per_pixel": layout.samples_per_pixel
        == int(gates["required_samples_per_pixel"]),
        "bits_per_sample": layout.bits_per_sample
        == tuple(
            int(gates["required_bits_per_sample"])
            for _ in range(layout.samples_per_pixel)
        ),
        "rows_per_strip": layout.rows_per_strip
        == int(gates["required_rows_per_strip"]),
        "endpoint_span": endpoint_span
        >= float(gates["minimum_endpoint_span_fraction"]),
        "horizontal_separator_count": len(edges)
        >= int(gates["minimum_high_confidence_horizontal_separators"]),
    }
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "source": {
            "archive_sha256": str(expected["archive_sha256"]),
            "archive_bytes": int(expected["archive_bytes"]),
            "member_path": str(expected["member_path"]),
            "member_crc32": str(expected["member_crc32"]),
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
            "central_profile_sha256": _array_sha256(profile),
            "top_endpoint_code": top,
            "bottom_endpoint_code": bottom,
            "endpoint_span_fraction": endpoint_span,
            "high_confidence_horizontal_separators": edges,
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "curve_fitting_performed": False,
        "decision": (
            "source_geometry_pass_open_separate_response_feasibility"
            if all(checks.values())
            else "close_source_geometry_without_curve_fitting"
        ),
        "development_disclosure": dict(config["development_disclosure"]),
        "claim_ceiling": str(config["claim_ceiling"]),
    }
    stable["stable_evidence_id"] = _sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return stable, sampled


def render_preview(sampled_u16: np.ndarray, path: Path) -> None:
    """Render a bounded display preview for source-geometry review only."""

    values = np.asarray(sampled_u16, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ApolloStepChartPreflightError("preview requires sampled RGB")
    low = np.percentile(values, 0.5, axis=(0, 1))
    high = np.percentile(values, 99.5, axis=(0, 1))
    if np.any(high <= low):
        raise ApolloStepChartPreflightError("preview percentile range collapsed")
    encoded = np.clip((values - low) / (high - low), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(encoded * 255.0).astype(np.uint8), "RGB").save(path)


def write_report(path: Path, report: Mapping[str, Any]) -> str:
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
    return _sha256(encoded)


__all__ = [
    "ApolloStepChartPreflightError",
    "evaluate_apollo_step_chart_preflight",
    "high_confidence_negative_edges",
    "render_preview",
    "validate_contract",
    "write_report",
]
