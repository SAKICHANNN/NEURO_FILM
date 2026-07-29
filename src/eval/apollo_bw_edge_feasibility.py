"""P5E feasibility audit for the AS16-111 camera-free B&W boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.apollo_step_chart_source import sha256_file
from src.eval.classic_tiff_stream import read_classic_tiff_zip_roi


SCHEMA = "neuro_film.u6_p5e_apollo_bw_edge_feasibility.v1"
REPORT_SCHEMA = "neuro_film.u6_p5e_apollo_bw_edge_feasibility_report.v1"


class ApolloBWEdgeFeasibilityError(ValueError):
    """Raised when the P5E contract or edge geometry fails closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(array).tobytes(order="C"))


def _crossing_position(profile: np.ndarray, midpoint: float) -> float | None:
    indices = np.flatnonzero(profile >= midpoint)
    if len(indices) == 0 or indices[0] == 0:
        return None
    right = int(indices[0])
    left = right - 1
    denominator = float(profile[right] - profile[left])
    if denominator <= 0.0:
        return None
    return float(left + (midpoint - profile[left]) / denominator)


def analyze_edge_roi(
    roi_u16: np.ndarray, analysis: Mapping[str, Any]
) -> dict[str, Any]:
    roi = np.asarray(roi_u16)
    if roi.ndim == 3 and roi.shape[2] == 1:
        roi = roi[..., 0]
    if (
        roi.ndim != 2
        or roi.dtype != np.uint16
        or min(roi.shape) <= 2 * int(analysis["plateau_rows"])
    ):
        raise ApolloBWEdgeFeasibilityError("invalid edge ROI")
    pixels = roi.astype(np.float64)
    plateau_rows = int(analysis["plateau_rows"])
    positions: list[tuple[int, float, float, float, float]] = []
    for column in range(pixels.shape[1]):
        profile = pixels[:, column]
        low = float(np.median(profile[:plateau_rows]))
        high = float(np.median(profile[-plateau_rows:]))
        span = high - low
        if span < float(analysis["minimum_column_span_code"]):
            continue
        crossing = _crossing_position(profile, (low + high) * 0.5)
        if crossing is not None:
            positions.append((column, crossing, low, high, span))
    if len(positions) < 2:
        raise ApolloBWEdgeFeasibilityError("insufficient valid edge columns")
    rows = np.asarray(positions, dtype=np.float64)
    mask = np.ones(len(rows), dtype=bool)
    for _ in range(int(analysis["robust_fit_iterations"])):
        if np.count_nonzero(mask) < 2:
            raise ApolloBWEdgeFeasibilityError("robust edge fit collapsed")
        coefficients = np.polyfit(rows[mask, 0], rows[mask, 1], 1)
        residual = rows[:, 1] - np.polyval(coefficients, rows[:, 0])
        median = float(np.median(residual[mask]))
        mad = float(np.median(np.abs(residual[mask] - median)))
        threshold = max(
            float(analysis["minimum_residual_pixels"]),
            float(analysis["mad_multiplier"]) * 1.4826 * mad,
        )
        mask = np.abs(residual - median) <= threshold
    inliers = rows[mask]
    phases = np.mod(inliers[:, 1], 1.0)
    phase_counts = np.histogram(
        phases,
        bins=int(analysis["phase_bin_count"]),
        range=(0.0, 1.0),
    )[0]
    oversampling = int(analysis["oversampling"])
    radius = int(analysis["profile_radius_pixels"])
    offsets = np.arange(
        -radius,
        radius + 1.0 / oversampling,
        1.0 / oversampling,
        dtype=np.float64,
    )
    profiles: list[np.ndarray] = []
    stride = int(analysis["profile_column_stride"])
    axis = np.arange(pixels.shape[0], dtype=np.float64)
    for column, crossing, low, high, _span in inliers[::stride]:
        query = crossing + offsets
        if query[0] < 0.0 or query[-1] > pixels.shape[0] - 1:
            continue
        normalized = (pixels[:, int(column)] - low) / (high - low)
        profiles.append(np.interp(query, axis, normalized))
    if not profiles:
        raise ApolloBWEdgeFeasibilityError("no complete aligned profiles")
    edge_spread = np.median(np.asarray(profiles), axis=0)

    def first_crossing(level: float) -> float:
        indices = np.flatnonzero(edge_spread >= level)
        if len(indices) == 0:
            raise ApolloBWEdgeFeasibilityError("edge spread misses level")
        return float(offsets[int(indices[0])])

    width_10_90 = first_crossing(0.9) - first_crossing(0.1)
    width_20_80 = first_crossing(0.8) - first_crossing(0.2)
    negative_fraction = float(
        np.mean(
            np.diff(edge_spread)
            < -float(analysis["negative_step_tolerance"])
        )
    )
    edge_shift = float(
        abs(coefficients[0]) * (pixels.shape[1] - 1)
    )
    return {
        "roi_shape": list(roi.shape),
        "roi_sha256": _array_sha256(roi),
        "valid_column_count": len(rows),
        "valid_column_fraction": float(len(rows) / pixels.shape[1]),
        "line_inlier_count": int(np.count_nonzero(mask)),
        "line_inlier_fraction": float(np.mean(mask)),
        "line_slope_y_per_x": float(coefficients[0]),
        "edge_shift_pixels": edge_shift,
        "inlier_residual_abs_p50": float(
            np.percentile(np.abs(residual[mask]), 50)
        ),
        "inlier_residual_abs_p95": float(
            np.percentile(np.abs(residual[mask]), 95)
        ),
        "phase_bin_counts": phase_counts.tolist(),
        "profile_count": len(profiles),
        "median_span_fraction": float(np.median(inliers[:, 4]) / 65535.0),
        "edge_spread_sha256": _array_sha256(edge_spread),
        "edge_spread_minimum": float(np.min(edge_spread)),
        "edge_spread_maximum": float(np.max(edge_spread)),
        "width_10_90_pixels": float(width_10_90),
        "width_20_80_pixels": float(width_20_80),
        "negative_step_fraction": negative_fraction,
    }


def validate_contract(config: Mapping[str, Any], root: Path) -> None:
    if config.get("schema") != SCHEMA:
        raise ApolloBWEdgeFeasibilityError("unsupported P5E contract")
    parent = config["parent"]
    decision_path = root / str(parent["decision_path"])
    raw = decision_path.read_bytes()
    if _sha256(raw) != str(parent["decision_sha256"]):
        raise ApolloBWEdgeFeasibilityError("P2L1 decision hash drifted")
    decision = json.loads(raw)
    if decision.get("decision") != parent["required_decision"]:
        raise ApolloBWEdgeFeasibilityError("P2L1 decision drifted")
    archive = root / str(parent["archive_path"])
    if (
        not archive.is_file()
        or sha256_file(archive) != parent["archive_sha256"]
    ):
        raise ApolloBWEdgeFeasibilityError("archive identity drifted")
    disclosure = config.get("development_disclosure", {})
    if not all(
        disclosure.get(key) is True
        for key in (
            "edge_probe_seen_before_numeric_gate_freeze",
            "observed_transition_wider_than_typical_sharp_edge",
            "untouched_confirmation_claim_forbidden",
        )
    ):
        raise ApolloBWEdgeFeasibilityError("development disclosure drifted")


def evaluate_apollo_bw_edge_feasibility(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    validate_contract(config, root)
    parent = config["parent"]
    roi_contract = config["roi"]
    _layout, roi = read_classic_tiff_zip_roi(
        root / str(parent["archive_path"]),
        str(parent["member_path"]),
        row_start=int(roi_contract["row_start"]),
        row_stop=int(roi_contract["row_stop"]),
        column_start=int(roi_contract["column_start"]),
        column_stop=int(roi_contract["column_stop"]),
        maximum_output_bytes=int(roi_contract["maximum_output_bytes"]),
        maximum_uncompressed_bytes=int(
            roi_contract["maximum_uncompressed_bytes"]
        ),
    )
    metrics = analyze_edge_roi(roi, config["edge_analysis"])
    gates = config["automatic_gates"]
    checks = {
        "valid_columns": metrics["valid_column_fraction"]
        >= float(gates["minimum_valid_column_fraction"]),
        "line_inliers": metrics["line_inlier_fraction"]
        >= float(gates["minimum_line_inlier_fraction"]),
        "edge_shift": metrics["edge_shift_pixels"]
        >= float(gates["minimum_edge_shift_pixels"]),
        "phase_coverage": min(metrics["phase_bin_counts"])
        >= int(gates["minimum_phase_bin_count"]),
        "span": metrics["median_span_fraction"]
        >= float(gates["minimum_median_span_fraction"]),
        "transition_width": metrics["width_10_90_pixels"]
        <= float(gates["maximum_10_90_width_pixels"]),
        "monotonicity": metrics["negative_step_fraction"]
        <= float(gates["maximum_negative_step_fraction"]),
    }
    automatic_pass = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "source": {
            "archive_sha256": str(parent["archive_sha256"]),
            "member_path": str(parent["member_path"]),
            "roi": dict(roi_contract),
        },
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_combined_edge_source_for_mtf_identifiability_only"
            if automatic_pass
            else "close_boundary_for_mtf_or_kernel_fitting"
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
    "ApolloBWEdgeFeasibilityError",
    "analyze_edge_roi",
    "evaluate_apollo_bw_edge_feasibility",
    "validate_contract",
    "write_report",
]
