"""U6.P4W label-blind fixed-crop scanner-code amplitude feasibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_nps import (
    _quadratic_detrend,
    fixed_fractional_crops,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import (
    hash_file,
    validate_contract as validate_source_contract,
)


SCHEMA = "neuro_film.u6_p4w_scan_amplitude_feasibility_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4w_scan_amplitude_feasibility_report.v1"


class ScanAmplitudeFeasibilityError(RuntimeError):
    """Raised when the P4W source, split or estimator drifts."""


def _load_exact_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise ScanAmplitudeFeasibilityError(f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScanAmplitudeFeasibilityError("parent must be a JSON object")
    return payload


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("parameter_fitting_allowed")
        or config["execution"].get("photographic_render_allowed")
        or config["execution"].get(
            "post_result_crop_or_threshold_change_allowed"
        )
        or config["split"].get("stock_labels_available_to_analysis")
        or config["split"].get("crop_pseudoreplication_allowed")
        or config["estimator"].get("content_based_crop_selection_allowed")
        or config["estimator"].get("outlier_or_dust_removal_allowed")
        or config["estimator"].get(
            "density_or_emulsion_interpretation_allowed"
        )
    ):
        raise ScanAmplitudeFeasibilityError("unsupported P4W contract")
    parents = config["parents"]
    decision = _load_exact_json(root, parents["p4v_decision"])
    source = _load_exact_json(root, parents["source_contract"])
    analysis = _load_exact_json(root, parents["analysis_contract"])
    execution = _load_exact_json(root, parents["execution_contract"])
    validate_source_contract(source)
    if (
        decision["decision"]
        != parents["p4v_decision"]["required_decision"]
        or execution["source_contract"]["sha256"]
        != parents["source_contract"]["sha256"]
        or execution["analysis_contract"]["sha256"]
        != parents["analysis_contract"]["sha256"]
    ):
        raise ScanAmplitudeFeasibilityError("P4W parent drift")
    all_ids = {
        str(row["title"])[5:-4] for row in source["files"]
    }
    development = config["split"]["development_source_ids"]
    confirmation = config["split"]["confirmation_source_ids"]
    if (
        len(development) != 4
        or len(confirmation) != 4
        or set(development) & set(confirmation)
        or set(development) | set(confirmation) != all_ids
    ):
        raise ScanAmplitudeFeasibilityError("P4W split drift")
    return source, analysis


def robust_relative_amplitude(crop: np.ndarray) -> float:
    values = np.asarray(crop, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) < 64
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("amplitude crop must be finite 2D")
    mean = float(np.mean(values, dtype=np.float64))
    if mean <= 0.0:
        raise ValueError("amplitude crop mean must be positive")
    residual = _quadratic_detrend(values / mean)
    center = float(np.median(residual))
    amplitude = 1.4826 * float(np.median(np.abs(residual - center)))
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ScanAmplitudeFeasibilityError("degenerate robust amplitude")
    return amplitude


def _coefficient_of_variation(values: list[float]) -> float:
    mean = float(np.mean(values, dtype=np.float64))
    if mean <= 0.0:
        raise ScanAmplitudeFeasibilityError("amplitude mean is not positive")
    return float(np.std(values, dtype=np.float64) / mean)


def evaluate_scan_amplitude_feasibility(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    source, analysis = validate_contract(root, config)
    pixel = analysis["pixel_contract"]
    rows = []
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        )
        del infrared
        channel_amplitudes = []
        for channel in range(3):
            channel_amplitudes.append(
                [
                    robust_relative_amplitude(crop)
                    for crop in fixed_fractional_crops(
                        rgb[..., channel],
                        crop_size=int(pixel["crop_size_pixels"]),
                        centers_yx=pixel["fixed_fractional_centers_yx"],
                    )
                ]
            )
        del rgb
        crop_amplitudes = [
            float(
                np.median(
                    [
                        channel_amplitudes[channel][crop_index]
                        for channel in range(3)
                    ]
                )
            )
            for crop_index in range(
                len(pixel["fixed_fractional_centers_yx"])
            )
        ]
        rows.append(
            {
                "source_id": inspected["source_id"],
                "source_sha256": inspected["sha256"],
                "channel_crop_amplitudes": channel_amplitudes,
                "crop_amplitudes": crop_amplitudes,
                "scan_amplitude": float(np.median(crop_amplitudes)),
                "within_scan_crop_coefficient_of_variation": (
                    _coefficient_of_variation(crop_amplitudes)
                ),
            }
        )
    by_id = {row["source_id"]: row for row in rows}
    if (
        len(rows) != int(config["automatic_gates"]["minimum_scan_groups"])
        or len(by_id) != len(rows)
    ):
        raise ScanAmplitudeFeasibilityError("P4W scan population drift")
    development = [
        by_id[source_id]["scan_amplitude"]
        for source_id in config["split"]["development_source_ids"]
    ]
    confirmation = [
        by_id[source_id]["scan_amplitude"]
        for source_id in config["split"]["confirmation_source_ids"]
    ]
    amplitudes = [row["scan_amplitude"] for row in rows]
    development_median = float(np.median(development))
    confirmation_median = float(np.median(confirmation))
    split_ratio = max(
        development_median / confirmation_median,
        confirmation_median / development_median,
    )
    summary = {
        "scan_count": len(rows),
        "minimum_scan_amplitude": float(min(amplitudes)),
        "maximum_scan_amplitude": float(max(amplitudes)),
        "median_scan_amplitude": float(np.median(amplitudes)),
        "maximum_within_scan_crop_coefficient_of_variation": float(
            max(
                row["within_scan_crop_coefficient_of_variation"]
                for row in rows
            )
        ),
        "across_scan_coefficient_of_variation": (
            _coefficient_of_variation(amplitudes)
        ),
        "across_scan_maximum_to_minimum_ratio": float(
            max(amplitudes) / min(amplitudes)
        ),
        "development_median_amplitude": development_median,
        "confirmation_median_amplitude": confirmation_median,
        "development_to_confirmation_median_ratio": float(split_ratio),
    }
    gates = config["automatic_gates"]
    checks = {
        "scan_groups": len(rows) >= int(gates["minimum_scan_groups"]),
        "within_scan_crop_repeatability": summary[
            "maximum_within_scan_crop_coefficient_of_variation"
        ]
        <= float(
            gates["maximum_within_scan_crop_coefficient_of_variation"]
        ),
        "across_scan_cv": summary["across_scan_coefficient_of_variation"]
        <= float(gates["maximum_across_scan_coefficient_of_variation"]),
        "across_scan_range": summary[
            "across_scan_maximum_to_minimum_ratio"
        ]
        <= float(gates["maximum_across_scan_maximum_to_minimum_ratio"]),
        "development_confirmation": summary[
            "development_to_confirmation_median_ratio"
        ]
        <= float(
            gates["maximum_development_to_confirmation_median_ratio"]
        ),
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "stock_labels_used": False,
        "rows": rows,
        "summary": summary,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_generic_same_scanner_amplitude_target"
            if automatic_pass
            else "close_scan_amplitude_fitting"
        ),
        "branch": config["branch_rule"][
            "pass" if automatic_pass else "fail"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_scan_amplitude_feasibility",
    "robust_relative_amplitude",
    "validate_contract",
    "write_report",
]
