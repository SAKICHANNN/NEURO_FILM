"""Effective edge-SFR temporal audit for the APPLAUSE TG13 scans.

The measured transition contains the target, scanner, and recorded workflow.
It is therefore an effective response and never an absolute scanner or film MTF.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.applause_tg13_temporal import open_fits_image, sha256_file

SCHEMA = "neuro_film.u6_p6ai_applause_effective_sfr_temporal_contract.v1"
LOCK_SCHEMA = "neuro_film.u6_p6ai_applause_effective_sfr_development_lock.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ai_applause_effective_sfr_temporal_report.v1"


class ApplauseEffectiveSFRError(RuntimeError):
    """Raised when the frozen effective-SFR contract fails closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApplauseEffectiveSFRError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ApplauseEffectiveSFRError("JSON input must be an object")
    return value


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    if config.get("schema") != SCHEMA:
        raise ApplauseEffectiveSFRError("unsupported P6AI contract schema")
    for parent in config.get("parents", {}).values():
        path = root / str(parent["path"])
        if not path.is_file() or sha256_file(path) != str(parent["sha256"]):
            raise ApplauseEffectiveSFRError(f"parent identity drifted: {path}")
    source = config.get("source", {})
    expected_roles = {
        "development_early": 3,
        "development_late": 5,
        "confirmation_early": 3,
        "confirmation_late": 3,
    }
    for role, count in expected_roles.items():
        values = source.get(role)
        if not isinstance(values, list) or len(values) != count:
            raise ApplauseEffectiveSFRError("source role support drifted")
    all_ids = [int(scan_id) for role in expected_roles for scan_id in source[role]]
    if len(set(all_ids)) != 14:
        raise ApplauseEffectiveSFRError("source scan identifiers must be unique")
    measurement = config.get("measurement", {})
    edges = measurement.get("nominal_step_edges_x")
    frequencies = measurement.get("curve_frequency_samples_cycles_per_pixel")
    if (
        not isinstance(edges, list)
        or len(edges) != 13
        or edges != sorted(edges)
        or not isinstance(frequencies, list)
        or frequencies != sorted(frequencies)
        or frequencies[0] <= 0.0
        or frequencies[-1] > 0.5
        or float(measurement.get("projected_bin_width_px", 0.0)) != 0.25
    ):
        raise ApplauseEffectiveSFRError("measurement geometry drifted")
    controls = config.get("controls", {})
    if controls.get("require_parent_file_hashes") is not True:
        raise ApplauseEffectiveSFRError("parent file hashes must remain required")
    if controls.get("require_parent_roles_exact") is not True:
        raise ApplauseEffectiveSFRError("parent roles must remain exact")
    if config.get("training_allowed") is not False:
        raise ApplauseEffectiveSFRError("training must remain forbidden")
    if config.get("product_integration_allowed") is not False:
        raise ApplauseEffectiveSFRError("product integration must remain forbidden")


def _parent_state(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    p6k_contract = _load_json(root / str(config["parents"]["p6k_contract"]["path"]))
    p6k_decision = _load_json(root / str(config["parents"]["p6k_decision"]["path"]))
    report_ref = p6k_decision.get("evidence", {}).get("report", {})
    report_path = root / str(report_ref.get("path", ""))
    if (
        p6k_decision.get("automatic_pass") is not True
        or not report_path.is_file()
        or sha256_file(report_path) != report_ref.get("sha256")
    ):
        raise ApplauseEffectiveSFRError("P6K evidence is unavailable or drifted")
    p6k_report = _load_json(report_path)
    rows = p6k_report.get("files")
    if not isinstance(rows, list) or len(rows) != 14:
        raise ApplauseEffectiveSFRError("P6K file evidence drifted")
    by_id = {int(row["scan_id"]): dict(row) for row in rows}
    if len(by_id) != 14 or any(len(str(row.get("sha256", ""))) != 64 for row in rows):
        raise ApplauseEffectiveSFRError("P6K file identities drifted")
    role_alias = {
        "development_early": "development_early",
        "development_late": "development_late",
        "confirmation_early": "confirmatory_early",
        "confirmation_late": "confirmatory_late",
    }
    for role, parent_role in role_alias.items():
        for scan_id in config["source"][role]:
            if by_id[int(scan_id)].get("role") != parent_role:
                raise ApplauseEffectiveSFRError("P6K role assignment drifted")
    return p6k_contract, p6k_report, by_id


def _pava_increasing(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    weights: list[int] = []
    for raw in np.asarray(values, dtype=np.float64):
        levels.append(float(raw))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
    return np.repeat(np.asarray(levels, dtype=np.float64), weights)


def _crossing(frequency: np.ndarray, values: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(values <= level)
    if indices.size == 0:
        return float(frequency[-1])
    right = int(indices[0])
    if right == 0:
        return float(frequency[0])
    left = right - 1
    denominator = float(values[right] - values[left])
    if denominator == 0.0:
        return float(frequency[right])
    return float(
        frequency[left]
        + (level - values[left]) * (frequency[right] - frequency[left]) / denominator
    )


def _sfr_from_esf(
    esf: np.ndarray,
    *,
    bin_width: float,
    samples: Sequence[float],
) -> tuple[np.ndarray, float]:
    lsf = np.diff(np.asarray(esf, dtype=np.float64))
    lsf = np.maximum(lsf, 0.0)
    total = float(np.sum(lsf))
    if not math.isfinite(total) or total <= 1e-12:
        raise ApplauseEffectiveSFRError("edge line spread has zero mass")
    lsf /= total
    spectrum = np.abs(np.fft.rfft(lsf, n=4096))
    spectrum /= spectrum[0]
    frequency = np.fft.rfftfreq(4096, d=bin_width)
    sample_array = np.asarray(samples, dtype=np.float64)
    curve = np.interp(sample_array, frequency, spectrum)
    nyquist = frequency <= 0.5
    mtf50 = _crossing(frequency[nyquist], spectrum[nyquist], 0.5)
    return curve, mtf50


def measure_edge_from_array(
    image: np.ndarray,
    *,
    nominal_edge_x: int,
    measurement: Mapping[str, Any],
    raw_code_span: float,
) -> dict[str, Any]:
    row_start = int(measurement["row_start"])
    row_stop = int(measurement["row_stop"])
    half_window = int(measurement["edge_half_window_px"])
    maximum_offset = float(measurement["maximum_row_edge_offset_px"])
    pixels = np.asarray(image)
    if pixels.ndim != 2 or not 0 <= row_start < row_stop <= pixels.shape[0]:
        raise ApplauseEffectiveSFRError("invalid edge source array")
    x_start = max(0, nominal_edge_x - half_window)
    x_stop = min(pixels.shape[1], nominal_edge_x + half_window + 1)
    roi = pixels[row_start:row_stop, x_start:x_stop].astype(np.float64)
    if roi.shape[1] < 7 or not np.isfinite(roi).all():
        raise ApplauseEffectiveSFRError("edge window is invalid")
    kernel = np.asarray(measurement["gradient_smoothing_kernel"], dtype=np.float64)
    if kernel.shape != (3,) or not np.isclose(float(np.sum(kernel)), 1.0):
        raise ApplauseEffectiveSFRError("gradient smoothing kernel drifted")
    smoothed = (
        kernel[0] * roi[:, :-2] + kernel[1] * roi[:, 1:-1] + kernel[2] * roi[:, 2:]
    )
    gradient = np.diff(smoothed, axis=1)
    gradient_x = x_start + 1.5 + np.arange(gradient.shape[1], dtype=np.float64)
    search = np.abs(gradient_x - float(nominal_edge_x)) <= maximum_offset
    if np.count_nonzero(search) < 2:
        raise ApplauseEffectiveSFRError("edge localization search is empty")
    search_indices = np.flatnonzero(search)
    local = np.abs(gradient[:, search])
    selected = search_indices[np.argmax(local, axis=1)]
    edge_x = gradient_x[selected]
    selected_gradient = gradient[np.arange(gradient.shape[0]), selected]
    polarity = 1.0 if float(np.median(selected_gradient)) >= 0.0 else -1.0
    y = np.arange(row_start, row_stop, dtype=np.float64)
    centered_y = y - float(np.mean(y))
    slope, intercept_centered = np.polyfit(centered_y, edge_x, 1)
    predicted = slope * centered_y + intercept_centered
    residual = edge_x - predicted
    inliers = np.abs(residual) <= float(measurement["robust_line_residual_px"])
    if np.count_nonzero(inliers) < int(measurement["minimum_line_inlier_rows"]):
        raise ApplauseEffectiveSFRError("edge line support is insufficient")
    slope, intercept_centered = np.polyfit(centered_y[inliers], edge_x[inliers], 1)
    intercept = float(intercept_centered - slope * float(np.mean(y)))
    residual = edge_x - (slope * y + intercept)
    inliers = np.abs(residual) <= float(measurement["robust_line_residual_px"])
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(measurement["minimum_line_inlier_rows"]):
        raise ApplauseEffectiveSFRError("refined edge line support is insufficient")

    radius = float(measurement["profile_radius_px"])
    bin_width = float(measurement["projected_bin_width_px"])
    bin_edges = np.arange(-radius, radius + bin_width * 1.01, bin_width)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    sample_y = y[inliers]
    sample_rows = pixels[row_start:row_stop][inliers].astype(np.float64)
    extent = math.ceil(radius + maximum_offset + 2)
    sample_x0 = max(0, nominal_edge_x - extent)
    sample_x1 = min(pixels.shape[1], nominal_edge_x + extent + 1)
    x = np.arange(sample_x0, sample_x1, dtype=np.float64)
    normalizer = math.sqrt(1.0 + float(slope) ** 2)
    distance = (
        x[None, :] - (float(slope) * sample_y[:, None] + intercept)
    ) / normalizer
    values = polarity * sample_rows[:, sample_x0:sample_x1]
    mask = np.abs(distance) <= radius
    indices = np.floor((distance[mask] + radius) / bin_width).astype(np.int64)
    indices = np.clip(indices, 0, centers.size - 1)
    flat_values = values[mask]
    esf = np.full(centers.size, np.nan, dtype=np.float64)
    for index in np.unique(indices):
        esf[int(index)] = float(np.median(flat_values[indices == index]))
    valid = np.isfinite(esf)
    if np.count_nonzero(valid) < centers.size // 4:
        raise ApplauseEffectiveSFRError("projected edge profile is too sparse")
    first = int(np.flatnonzero(valid)[0])
    last = int(np.flatnonzero(valid)[-1])
    if first > np.searchsorted(centers, -float(measurement["wing_start_px"])):
        raise ApplauseEffectiveSFRError("left edge wing is unsupported")
    if last < np.searchsorted(centers, float(measurement["wing_start_px"])):
        raise ApplauseEffectiveSFRError("right edge wing is unsupported")
    esf = np.interp(centers, centers[valid], esf[valid])
    wing = float(measurement["wing_start_px"])
    left = float(np.median(esf[centers <= -wing]))
    right = float(np.median(esf[centers >= wing]))
    span = right - left
    if not math.isfinite(span) or span <= 0.0 or raw_code_span <= 0.0:
        raise ApplauseEffectiveSFRError("edge contrast is invalid")
    normalized_contrast = span / float(raw_code_span)
    normalized = _pava_increasing((esf - left) / span)
    normalized = np.clip(normalized, 0.0, 1.0)
    curve, mtf50 = _sfr_from_esf(
        normalized,
        bin_width=bin_width,
        samples=measurement["curve_frequency_samples_cycles_per_pixel"],
    )
    return {
        "nominal_edge_x": int(nominal_edge_x),
        "line_slope_x_per_y": float(slope),
        "line_inlier_rows": inlier_count,
        "line_residual_abs_p95_px": float(np.percentile(np.abs(residual[inliers]), 95)),
        "normalized_contrast": float(normalized_contrast),
        "profile_observed_bins": int(np.count_nonzero(valid)),
        "sfr_curve": [float(value) for value in curve],
        "mtf50_cycles_per_pixel": float(mtf50),
    }


def _measure_scan(
    path: Path,
    *,
    p6k_contract: Mapping[str, Any],
    parent_row: Mapping[str, Any],
    measurement: Mapping[str, Any],
    selected_edges: set[int] | None,
) -> dict[str, Any]:
    expected_sha = str(parent_row["sha256"])
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ApplauseEffectiveSFRError(f"scan file identity drifted: {path}")
    image, _header = open_fits_image(path, p6k_contract["fits"])
    raw_span = float(parent_row["raw_code_span"])
    edge_rows: list[dict[str, Any]] = []
    minimum_contrast = float(
        measurement["development_edge_minimum_normalized_contrast"]
    )
    for edge_index, nominal_x in enumerate(measurement["nominal_step_edges_x"]):
        if selected_edges is not None and edge_index not in selected_edges:
            continue
        try:
            result = measure_edge_from_array(
                image,
                nominal_edge_x=int(nominal_x),
                measurement=measurement,
                raw_code_span=raw_span,
            )
            result["edge_index"] = edge_index
            result["eligible"] = result["normalized_contrast"] >= minimum_contrast
            result["failure"] = None
        except ApplauseEffectiveSFRError as exc:
            if selected_edges is not None:
                raise
            result = {
                "edge_index": edge_index,
                "nominal_edge_x": int(nominal_x),
                "eligible": False,
                "failure": str(exc),
            }
        edge_rows.append(result)
    del image
    if selected_edges is not None and len(edge_rows) != len(selected_edges):
        raise ApplauseEffectiveSFRError("selected edge support drifted")
    eligible = [row for row in edge_rows if row["eligible"]]
    aggregate: dict[str, Any] | None = None
    if eligible:
        aggregate = {
            "sfr_curve": [
                float(value)
                for value in np.median(
                    np.asarray(
                        [row["sfr_curve"] for row in eligible], dtype=np.float64
                    ),
                    axis=0,
                )
            ],
            "mtf50_cycles_per_pixel": float(
                np.median([row["mtf50_cycles_per_pixel"] for row in eligible])
            ),
        }
    return {
        "sha256": expected_sha,
        "raw_code_span": raw_span,
        "edges": edge_rows,
        "aggregate": aggregate,
    }


def _rmse(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(left - right))))


def build_development_lock(
    root: Path,
    config: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    validate_config(config, root)
    p6k_contract, _p6k_report, by_id = _parent_state(root, config)
    measurement = config["measurement"]
    development: list[dict[str, Any]] = []
    for regime in ("early", "late"):
        role = f"development_{regime}"
        for scan_id in config["source"][role]:
            path = (
                root
                / str(config["source"]["parent_data_root"])
                / f"scan_{int(scan_id):06d}.fits"
            )
            scan = _measure_scan(
                path,
                p6k_contract=p6k_contract,
                parent_row=by_id[int(scan_id)],
                measurement=measurement,
                selected_edges=None,
            )
            development.append({"scan_id": int(scan_id), "regime": regime, **scan})
    eligible_sets = [
        {int(row["edge_index"]) for row in scan["edges"] if row["eligible"]}
        for scan in development
    ]
    shared_edges = sorted(set.intersection(*eligible_sets))
    development_gate_pass = len(shared_edges) >= int(
        measurement["minimum_shared_eligible_edges"]
    )
    prototypes: dict[str, dict[str, Any]] = {}
    if development_gate_pass:
        for scan in development:
            rows = [
                row for row in scan["edges"] if int(row["edge_index"]) in shared_edges
            ]
            scan["aggregate"] = {
                "sfr_curve": [
                    float(value)
                    for value in np.median(
                        np.asarray(
                            [row["sfr_curve"] for row in rows], dtype=np.float64
                        ),
                        axis=0,
                    )
                ],
                "mtf50_cycles_per_pixel": float(
                    np.median([row["mtf50_cycles_per_pixel"] for row in rows])
                ),
            }
        for regime in ("early", "late"):
            rows = [scan for scan in development if scan["regime"] == regime]
            prototypes[regime] = {
                "sfr_curve": [
                    float(value)
                    for value in np.median(
                        np.asarray([row["aggregate"]["sfr_curve"] for row in rows]),
                        axis=0,
                    )
                ],
                "mtf50_cycles_per_pixel": float(
                    np.median(
                        [row["aggregate"]["mtf50_cycles_per_pixel"] for row in rows]
                    )
                ),
            }
        prototypes["shared"] = {
            "sfr_curve": [
                float(value)
                for value in np.median(
                    np.asarray([row["aggregate"]["sfr_curve"] for row in development]),
                    axis=0,
                )
            ],
            "mtf50_cycles_per_pixel": float(
                np.median(
                    [row["aggregate"]["mtf50_cycles_per_pixel"] for row in development]
                )
            ),
        }
    body = {
        "schema": LOCK_SCHEMA,
        "experiment_id": config["experiment_id"],
        "contract_sha256": contract_sha256,
        "frequency_samples_cycles_per_pixel": list(
            measurement["curve_frequency_samples_cycles_per_pixel"]
        ),
        "selected_edge_indices": shared_edges,
        "minimum_required_shared_edges": int(
            measurement["minimum_shared_eligible_edges"]
        ),
        "development_gate_pass": development_gate_pass,
        "development_scans": development,
        "prototypes": prototypes,
        "algorithm": "robust-line/projected-quarter-pixel-median/pava/nonnegative-lsf-rfft4096-v1",
    }
    return {**body, "development_lock_id": _sha256_bytes(_canonical_bytes(body))}


def validate_development_lock(
    lock: Mapping[str, Any], config: Mapping[str, Any], *, contract_sha256: str
) -> None:
    if lock.get("schema") != LOCK_SCHEMA or lock.get("experiment_id") != config.get(
        "experiment_id"
    ):
        raise ApplauseEffectiveSFRError("development lock identity drifted")
    if lock.get("contract_sha256") != contract_sha256:
        raise ApplauseEffectiveSFRError("development lock contract drifted")
    body = dict(lock)
    lock_id = body.pop("development_lock_id", None)
    if lock_id != _sha256_bytes(_canonical_bytes(body)):
        raise ApplauseEffectiveSFRError("development lock hash drifted")
    selected = lock.get("selected_edge_indices")
    minimum = int(config["measurement"]["minimum_shared_eligible_edges"])
    if not isinstance(selected, list) or selected != sorted(
        {int(value) for value in selected}
    ):
        raise ApplauseEffectiveSFRError("development edge selection drifted")
    if lock.get("minimum_required_shared_edges") != minimum or lock.get(
        "development_gate_pass"
    ) != (len(selected) >= minimum):
        raise ApplauseEffectiveSFRError("development edge gate drifted")


def build_report_from_lock(
    root: Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    validate_config(config, root)
    validate_development_lock(lock, config, contract_sha256=contract_sha256)
    p6k_contract, _p6k_report, by_id = _parent_state(root, config)
    selected = {int(value) for value in lock["selected_edge_indices"]}
    if lock["development_gate_pass"] is False:
        stable_body = {
            "schema": REPORT_SCHEMA,
            "experiment_id": config["experiment_id"],
            "contract_sha256": contract_sha256,
            "development_lock_id": lock["development_lock_id"],
            "selected_edge_indices": list(lock["selected_edge_indices"]),
            "development_eligible_edge_counts": {
                str(row["scan_id"]): sum(edge["eligible"] for edge in row["edges"])
                for row in lock["development_scans"]
            },
            "minimum_required_shared_edges": lock["minimum_required_shared_edges"],
            "confirmation_file_reads": 0,
            "shared_stability_pass": False,
            "regime_nuisance_pass": False,
            "automatic_pass": False,
            "decision": "close_before_confirmation_insufficient_shared_edges",
            "claim_ceiling": config["claim_ceiling"],
        }
        return {
            **stable_body,
            "stable_evidence_id": _sha256_bytes(_canonical_bytes(stable_body)),
        }
    confirmation: list[dict[str, Any]] = []
    for regime in ("early", "late"):
        role = f"confirmation_{regime}"
        for scan_id in config["source"][role]:
            path = (
                root
                / str(config["source"]["parent_data_root"])
                / f"scan_{int(scan_id):06d}.fits"
            )
            scan = _measure_scan(
                path,
                p6k_contract=p6k_contract,
                parent_row=by_id[int(scan_id)],
                measurement=config["measurement"],
                selected_edges=selected,
            )
            if scan["aggregate"] is None or any(
                not row["eligible"] for row in scan["edges"]
            ):
                raise ApplauseEffectiveSFRError(
                    "confirmation edge failed frozen support"
                )
            correct = lock["prototypes"][regime]
            wrong = lock["prototypes"]["late" if regime == "early" else "early"]
            shared = lock["prototypes"]["shared"]
            correct_rmse = _rmse(scan["aggregate"]["sfr_curve"], correct["sfr_curve"])
            wrong_rmse = _rmse(scan["aggregate"]["sfr_curve"], wrong["sfr_curve"])
            shared_rmse = _rmse(scan["aggregate"]["sfr_curve"], shared["sfr_curve"])
            mtf50_drift = abs(
                float(scan["aggregate"]["mtf50_cycles_per_pixel"])
                - float(shared["mtf50_cycles_per_pixel"])
            ) / max(abs(float(shared["mtf50_cycles_per_pixel"])), 1e-12)
            confirmation.append(
                {
                    "scan_id": int(scan_id),
                    "regime": regime,
                    **scan,
                    "correct_prototype_rmse": correct_rmse,
                    "wrong_prototype_rmse": wrong_rmse,
                    "wrong_minus_correct_rmse": wrong_rmse - correct_rmse,
                    "shared_prototype_rmse": shared_rmse,
                    "relative_mtf50_drift_from_shared": float(mtf50_drift),
                }
            )
    development_rmse = _rmse(
        lock["prototypes"]["early"]["sfr_curve"],
        lock["prototypes"]["late"]["sfr_curve"],
    )
    gates = config["gates"]
    shared_gates = gates["shared_stability"]
    regime_gates = gates["regime_nuisance"]
    correct_wins = sum(
        row["correct_prototype_rmse"] < row["wrong_prototype_rmse"]
        for row in confirmation
    )
    shared_checks = {
        "confirmation_support": len(confirmation)
        == int(shared_gates["required_confirmation_rows"]),
        "development_regime_curve_rmse": development_rmse
        <= float(
            shared_gates["maximum_development_regime_curve_rmse_for_shared_claim"]
        ),
        "confirmation_curve_rmse": max(
            row["shared_prototype_rmse"] for row in confirmation
        )
        <= float(
            shared_gates[
                "maximum_confirmation_curve_rmse_to_shared_development_prototype"
            ]
        ),
        "confirmation_mtf50_drift": max(
            row["relative_mtf50_drift_from_shared"] for row in confirmation
        )
        <= float(shared_gates["maximum_confirmation_relative_mtf50_drift"]),
    }
    regime_checks = {
        "development_regime_curve_rmse": development_rmse
        >= float(regime_gates["minimum_development_early_late_curve_rmse"]),
        "correct_prototype_wins": correct_wins
        == int(regime_gates["required_correct_prototype_wins"]),
        "confirmation_margin": min(
            row["wrong_minus_correct_rmse"] for row in confirmation
        )
        >= float(regime_gates["minimum_confirmation_wrong_minus_correct_curve_rmse"]),
    }
    regime_pass = all(regime_checks.values())
    shared_pass = all(shared_checks.values())
    if regime_pass:
        decision = "retain_effective_sfr_as_time_regime_scanner_nuisance"
    elif shared_pass:
        decision = "open_same_workflow_effective_sfr_compiler"
    else:
        decision = "close_effective_sfr_temporal_route"
    stable_body = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "contract_sha256": contract_sha256,
        "development_lock_id": lock["development_lock_id"],
        "selected_edge_indices": list(lock["selected_edge_indices"]),
        "development_prototypes": lock["prototypes"],
        "development_regime_curve_rmse": development_rmse,
        "confirmation_scans": confirmation,
        "metrics": {
            "correct_prototype_wins": correct_wins,
            "minimum_wrong_minus_correct_curve_rmse": min(
                row["wrong_minus_correct_rmse"] for row in confirmation
            ),
            "maximum_confirmation_shared_curve_rmse": max(
                row["shared_prototype_rmse"] for row in confirmation
            ),
            "maximum_confirmation_relative_mtf50_drift": max(
                row["relative_mtf50_drift_from_shared"] for row in confirmation
            ),
        },
        "shared_checks": shared_checks,
        "regime_checks": regime_checks,
        "shared_stability_pass": shared_pass,
        "regime_nuisance_pass": regime_pass,
        "automatic_pass": regime_pass or shared_pass,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable_body,
        "stable_evidence_id": _sha256_bytes(_canonical_bytes(stable_body)),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
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
    "ApplauseEffectiveSFRError",
    "build_development_lock",
    "build_report_from_lock",
    "measure_edge_from_array",
    "validate_config",
    "validate_development_lock",
    "write_json",
]
