"""Execute the preregistered real uniform-grain NPS/ACF feasibility audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_nps import (
    acf_lag_signature,
    evaluate_signatures,
    fixed_fractional_crops,
    radial_nps_signature,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file, validate_contract


class UniformGrainAnalysisError(RuntimeError):
    """Raised when analysis admission or source identity fails closed."""


def _load_bound_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise UniformGrainAnalysisError(f"evidence hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UniformGrainAnalysisError("bound evidence must be an object")
    return payload


def build_scan_signatures(
    *,
    rgb: np.ndarray,
    infrared: np.ndarray,
    pixel_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build fixed-position RGB and IR signatures without crop selection."""
    crop_size = int(pixel_contract["crop_size_pixels"])
    centers = pixel_contract["fixed_fractional_centers_yx"]
    edges = pixel_contract["radial_nps_band_edges_cycles_per_pixel"]
    lags = pixel_contract["acf_lags_pixels_yx"]
    channel_rows: dict[str, list[np.ndarray]] = {
        "red": [],
        "green": [],
        "blue": [],
        "infrared": [],
    }
    channel_acf: dict[str, list[np.ndarray]] = {
        key: [] for key in channel_rows
    }
    names = ("red", "green", "blue")
    for channel, name in enumerate(names):
        for crop in fixed_fractional_crops(
            rgb[..., channel],
            crop_size=crop_size,
            centers_yx=centers,
        ):
            channel_rows[name].append(
                radial_nps_signature(
                    crop,
                    band_edges_cycles_per_pixel=edges,
                )
            )
            channel_acf[name].append(
                acf_lag_signature(crop, lags_yx=lags)
            )
    for crop in fixed_fractional_crops(
        infrared,
        crop_size=crop_size,
        centers_yx=centers,
    ):
        channel_rows["infrared"].append(
            radial_nps_signature(
                crop,
                band_edges_cycles_per_pixel=edges,
            )
        )
        channel_acf["infrared"].append(
            acf_lag_signature(crop, lags_yx=lags)
        )
    combined = []
    for crop_index in range(len(centers)):
        row = np.concatenate(
            [
                channel_rows[name][crop_index]
                for name in names
            ]
        )
        row /= np.linalg.norm(row)
        combined.append(row)
    return {
        "combined_rgb_nps": np.asarray(combined),
        "channel_nps": {
            name: np.asarray(rows) for name, rows in channel_rows.items()
        },
        "channel_acf": {
            name: np.asarray(rows) for name, rows in channel_acf.items()
        },
    }


def run_analysis(
    *,
    root: Path,
    source_config: dict[str, Any],
    analysis_config: dict[str, Any],
    execution_config: dict[str, Any],
) -> dict[str, Any]:
    """Run scan-group evaluation only after exact preflight/visual admission."""
    validate_contract(source_config)
    if (
        execution_config.get("schema")
        != "neuro_film.u6_p4r_uniform_grain_nps_execution.v1"
    ):
        raise UniformGrainAnalysisError("unsupported execution schema")
    bound_source = _load_bound_json(
        root,
        execution_config["source_contract"],
    )
    bound_analysis = _load_bound_json(
        root,
        execution_config["analysis_contract"],
    )
    if bound_source != source_config or bound_analysis != analysis_config:
        raise UniformGrainAnalysisError("execution contract binding mismatch")
    preflight = _load_bound_json(
        root,
        execution_config["preflight_report"],
    )
    visual = _load_bound_json(
        root,
        execution_config["visual_review"],
    )
    expected_review_ids = {
        str(row["title"])[5:-4] for row in source_config["files"]
    }
    if (
        preflight.get("schema")
        != "neuro_film.u6_p4r_uniform_grain_preflight_report.v1"
        or preflight.get("source_count") != 8
        or not preflight.get("all_exact_integrity")
        or not preflight.get("all_single_page_uint16_rgba")
        or visual.get("schema")
        != "neuro_film.u6_p4r_uniform_grain_visual_review.v1"
        or not visual.get("passed")
        or visual.get("confirmed_severe_source_artifact_count") != 0
        or set(visual.get("reviewed_source_ids", []))
        != expected_review_ids
    ):
        raise UniformGrainAnalysisError("pixel/visual admission did not pass")
    scan_ids: list[str] = []
    stock_ids: list[str] = []
    combined: list[np.ndarray] = []
    per_channel: dict[str, list[np.ndarray]] = {
        name: [] for name in ("red", "green", "blue", "infrared")
    }
    acf_rows: dict[str, list[list[float]]] = {
        name: [] for name in per_channel
    }
    source_rows: list[dict[str, Any]] = []
    for expected in sorted(
        source_config["files"],
        key=lambda row: str(row["title"]),
    ):
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / str(expected["path"]),
            expected=expected,
            crop_size=int(
                analysis_config["pixel_contract"]["crop_size_pixels"]
            ),
            centers_yx=analysis_config["pixel_contract"][
                "fixed_fractional_centers_yx"
            ],
        )
        signatures = build_scan_signatures(
            rgb=rgb,
            infrared=infrared,
            pixel_contract=analysis_config["pixel_contract"],
        )
        del rgb, infrared
        scan_ids.append(inspected["source_id"])
        stock_ids.append(str(expected["film_stock_id"]))
        combined.append(signatures["combined_rgb_nps"])
        for name in per_channel:
            per_channel[name].append(signatures["channel_nps"][name])
            acf_rows[name].append(
                np.median(signatures["channel_acf"][name], axis=0).tolist()
            )
        source_rows.append(
            {
                "source_id": inspected["source_id"],
                "film_stock_id": expected["film_stock_id"],
                "sha256": inspected["sha256"],
                "path": expected["path"],
            }
        )
    gates = analysis_config["automatic_gates"]
    primary = evaluate_signatures(
        scan_ids=scan_ids,
        stock_ids=stock_ids,
        crop_signatures=combined,
        gates=gates,
    )
    channel_results = {
        name: evaluate_signatures(
            scan_ids=scan_ids,
            stock_ids=stock_ids,
            crop_signatures=rows,
            gates=gates,
        )
        for name, rows in per_channel.items()
    }
    stable_payload = {
        "schema": "neuro_film.u6_p4r_uniform_grain_nps_report.v1",
        "source_contract_sha256": hash_file(
            root / "configs/u6_p4r_uniform_grain_source_v1.json",
            "sha256",
        ),
        "analysis_contract_sha256": hash_file(
            root / "configs/u6_p4r_uniform_grain_nps_feasibility_v1.json",
            "sha256",
        ),
        "preflight_report_sha256": execution_config[
            "preflight_report"
        ]["sha256"],
        "visual_review_sha256": execution_config["visual_review"]["sha256"],
        "source_rows": source_rows,
        "primary_combined_rgb_nps": primary,
        "channel_nps_results": channel_results,
        "median_scan_acf_by_channel": acf_rows,
        "parameter_fitting_performed": False,
        "decision": primary["branch"],
        "claim_ceiling": analysis_config["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {**stable_payload, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "UniformGrainAnalysisError",
    "build_scan_signatures",
    "run_analysis",
    "write_report",
]
