"""RF3.D5 normalized comparison of first-party spectral-sensitivity shapes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCHEMA = "neuro-film.rf3-three-stock-spectral-shape-contract.v1"
DIGITIZATION_SCHEMA = "neuro-film.rf3-three-stock-spectral-digitization.v1"
REPORT_SCHEMA = "neuro-film.rf3-three-stock-spectral-shape-report.v1"
STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]


class ThreeStockSpectralShapeError(RuntimeError):
    """Raised when the frozen spectral-shape experiment drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockSpectralShapeError("RF3.D5 paths must be repository-relative")
    return path


def _linear(value: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    return float(pixel0 + (value - value0) * (pixel1 - pixel0) / (value1 - value0))


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digitization = payload.get("digitization", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "RF3.D5"
        or [row.get("stock_id") for row in payload.get("sources", [])] != STOCKS
        or payload.get("wavelengths_nm") != list(range(400, 681, 10))
        or digitization.get("outside_published_curve_support") != -1.25
        or digitization.get("minimum_points_per_channel") != 9
        or gates.get(
            "minimum_pairwise_active_union_mean_absolute_log_shape_difference_lower_bound"
        )
        != 0.05
        or gates.get(
            "minimum_pairwise_active_union_rmse_log_shape_difference_lower_bound"
        )
        != 0.08
        or gates.get("minimum_pairwise_maximum_channel_peak_wavelength_difference_nm")
        != 10.0
        or not all(
            gates.get(key) is True
            for key in (
                "require_all_pairwise_gates",
                "require_all_source_hashes",
                "require_exact_replay",
            )
        )
    ):
        raise ThreeStockSpectralShapeError("RF3.D5 frozen contract drift")
    _relative(payload["parent"]["path"])
    _relative(digitization["observation_file"])
    for source in payload["sources"]:
        _relative(source["page_image"])
    return payload


def _load_digitization(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], str]:
    path = root / _relative(contract["digitization"]["observation_file"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != DIGITIZATION_SCHEMA
        or [row.get("stock_id") for row in payload.get("stocks", [])] != STOCKS
    ):
        raise ThreeStockSpectralShapeError("RF3.D5 digitization identity drift")
    channels = contract["channels"]
    minimum = int(contract["digitization"]["minimum_points_per_channel"])
    for stock in payload["stocks"]:
        if list(stock.get("curves", {})) != channels:
            raise ThreeStockSpectralShapeError("RF3.D5 channel order drift")
        for points in stock["curves"].values():
            values = np.asarray(points, dtype=np.float64)
            if (
                values.ndim != 2
                or values.shape[1] != 2
                or len(values) < minimum
                or not np.all(np.isfinite(values))
                or not np.all(np.diff(values[:, 0]) > 0)
            ):
                raise ThreeStockSpectralShapeError("RF3.D5 digitization points invalid")
    return payload, _hash_file(path)


def evaluate(
    contract: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    parent = root / _relative(contract["parent"]["path"])
    if _hash_file(parent) != contract["parent"]["sha256"]:
        raise ThreeStockSpectralShapeError("RF3.D5 parent evidence drift")
    if (
        json.loads(parent.read_text(encoding="utf-8")).get("decision")
        != contract["parent"]["required_decision"]
    ):
        raise ThreeStockSpectralShapeError("RF3.D5 parent decision drift")
    digitization, digitization_sha = _load_digitization(contract, root)
    by_stock = {row["stock_id"]: row for row in digitization["stocks"]}
    wavelength = np.asarray(contract["wavelengths_nm"], dtype=np.float64)
    floor = float(contract["digitization"]["outside_published_curve_support"])
    traces: list[dict[str, Any]] = []
    overlay_hashes: dict[str, str] = {}
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for source in contract["sources"]:
        page = root / _relative(source["page_image"])
        if _hash_file(page) != source["page_image_sha256"]:
            raise ThreeStockSpectralShapeError("RF3.D5 page image drift")
        image = cv2.imread(str(page), cv2.IMREAD_COLOR)
        if (
            image is None
            or [image.shape[1], image.shape[0]] != source["page_image_size"]
        ):
            raise ThreeStockSpectralShapeError("RF3.D5 page geometry drift")
        curves = []
        for channel in contract["channels"]:
            points = np.asarray(
                by_stock[source["stock_id"]]["curves"][channel], dtype=np.float64
            )
            sampled = np.interp(
                wavelength, points[:, 0], points[:, 1], left=-math.inf, right=-math.inf
            )
            normalized = np.maximum(sampled - float(np.max(points[:, 1])), floor)
            peak_wavelength = float(points[int(np.argmax(points[:, 1])), 0])
            curves.append(
                {
                    "channel": channel,
                    "peak_wavelength_nm": peak_wavelength,
                    "normalized_log_shape": normalized.tolist(),
                }
            )
            polyline = np.asarray(
                [
                    [
                        _linear(row[0], source["x_wavelength_anchors"]),
                        _linear(row[1], source["y_log_sensitivity_anchors"]),
                    ]
                    for row in points
                ],
                dtype=np.int32,
            )
            cv2.polylines(image, [polyline], False, (0, 0, 255), 2)
            for point in polyline:
                cv2.circle(image, tuple(point), 4, (255, 0, 0), 1)
        overlay_path = overlay_dir / f"{source['stock_id']}_overlay.png"
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(overlay_path)
        overlay_hashes[source["stock_id"]] = _hash_file(overlay_path)
        y0, y1 = source["y_log_sensitivity_anchors"]
        uncertainty = (
            abs(float(y1[0]) - float(y0[0]))
            / abs(float(y1[1]) - float(y0[1]))
            * float(source["digitization_maximum_y_uncertainty_pixels"])
        )
        traces.append(
            {
                "stock_id": source["stock_id"],
                "film_family": source["film_family"],
                "published_semantics": source["published_semantics"],
                "maximum_log_sensitivity_uncertainty": uncertainty,
                "curves": curves,
            }
        )

    med_gate = float(
        contract["gates"][
            "minimum_pairwise_active_union_mean_absolute_log_shape_difference_lower_bound"
        ]
    )
    rmse_gate = float(
        contract["gates"][
            "minimum_pairwise_active_union_rmse_log_shape_difference_lower_bound"
        ]
    )
    peak_gate = float(
        contract["gates"][
            "minimum_pairwise_maximum_channel_peak_wavelength_difference_nm"
        ]
    )
    comparisons = []
    for left, right in itertools.combinations(traces, 2):
        left_values = np.asarray(
            [row["normalized_log_shape"] for row in left["curves"]]
        )
        right_values = np.asarray(
            [row["normalized_log_shape"] for row in right["curves"]]
        )
        active = (left_values > floor) | (right_values > floor)
        delta = np.abs(left_values[active] - right_values[active])
        observed_mean = float(np.mean(delta))
        observed_rmse = float(np.sqrt(np.mean(np.square(delta))))
        uncertainty = float(
            left["maximum_log_sensitivity_uncertainty"]
            + right["maximum_log_sensitivity_uncertainty"]
        )
        mean_lower = max(0.0, observed_mean - uncertainty)
        rmse_lower = max(0.0, observed_rmse - uncertainty)
        peak_differences = [
            abs(float(a["peak_wavelength_nm"]) - float(b["peak_wavelength_nm"]))
            for a, b in zip(left["curves"], right["curves"], strict=True)
        ]
        comparisons.append(
            {
                "left": left["stock_id"],
                "right": right["stock_id"],
                "active_union_sample_count": int(np.sum(active)),
                "observed_mean_absolute_log_shape_difference": observed_mean,
                "observed_rmse_log_shape_difference": observed_rmse,
                "combined_uncertainty_penalty": uncertainty,
                "mean_absolute_lower_bound": mean_lower,
                "rmse_lower_bound": rmse_lower,
                "channel_peak_wavelength_differences_nm": peak_differences,
                "maximum_channel_peak_wavelength_difference_nm": max(peak_differences),
                "mean_gate_pass": mean_lower >= med_gate,
                "rmse_gate_pass": rmse_lower >= rmse_gate,
                "peak_gate_pass": max(peak_differences) >= peak_gate,
            }
        )
    gate_results = {
        "all_source_hashes_and_digitizations_valid": len(traces) == 3,
        "all_pairwise_mean_lower_bounds_material": all(
            row["mean_gate_pass"] for row in comparisons
        ),
        "all_pairwise_rmse_lower_bounds_material": all(
            row["rmse_gate_pass"] for row in comparisons
        ),
        "all_pairwise_peak_differences_material": all(
            row["peak_gate_pass"] for row in comparisons
        ),
    }
    automatic_pass = all(gate_results.values())
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "digitization_sha256": digitization_sha,
        "wavelengths_nm": contract["wavelengths_nm"],
        "traces": traces,
        "pairwise_comparisons": comparisons,
        "overlay_sha256": overlay_hashes,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = dict(result)
    stable.pop("overlay_sha256")
    result["stable_evidence_id"] = hashlib.sha256(_canonical_json(stable)).hexdigest()
    return result


def write_report(report: Mapping[str, Any], path: Path) -> str:
    data = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
