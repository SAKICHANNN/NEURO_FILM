"""RF3.D3 deterministic comparison of three published MTF graph envelopes."""

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

SCHEMA = "neuro-film.rf3-three-stock-mtf-prior-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-three-stock-mtf-prior-report.v1"


class ThreeStockMtfError(RuntimeError):
    """Raised when the frozen RF3.D3 contract or source raster drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ThreeStockMtfError("RF3.D3 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    trace = payload.get("trace", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "RF3.D3"
        or payload.get("frequencies_cycles_per_mm") != [8.0, 15.0, 30.0, 40.0, 60.0]
        or [row.get("stock_id") for row in sources]
        != ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
        or trace.get("black_threshold_uint8") != 90
        or trace.get("x_half_window_pixels") != 4
        or trace.get("minimum_black_hits_per_row") != 2
        or trace.get("horizontal_grid_mask_radius_pixels") != 3
        or trace.get("minimum_response_percent") != 15.0
        or trace.get("maximum_response_percent") != 140.0
        or gates.get("minimum_pairwise_median_absolute_log10_response_difference")
        != 0.03
        or gates.get("minimum_pairwise_rmse_log10_response_difference") != 0.04
        or not all(
            gates.get(key) is True
            for key in (
                "require_all_frequencies_detected",
                "require_source_semantics_reported",
                "exact_replay_required",
            )
        )
    ):
        raise ThreeStockMtfError("RF3.D3 frozen contract drift")
    _relative(str(payload.get("parent", {}).get("path", "")))
    for row in sources:
        _relative(str(row.get("page_image", "")))
        if row.get("expected_curve_count") not in (1, 3):
            raise ThreeStockMtfError("RF3.D3 invalid expected curve count")
    return payload


def _linear_from_anchors(value: float, anchors: Sequence[Sequence[float]]) -> float:
    (v0, p0), (v1, p1) = anchors
    return float(p0 + ((value - v0) / (v1 - v0)) * (p1 - p0))


def _value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (v0, p0), (v1, p1) = anchors
    return float(v0 + ((pixel - p0) / (p1 - p0)) * (v1 - v0))


def _clusters(values: Sequence[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    result: list[list[tuple[int, float]]] = []
    for item in values:
        if not result or item[0] > result[-1][-1][0] + 1:
            result.append([item])
        else:
            result[-1].append(item)
    return result


def _trace_source(
    source: Mapping[str, Any], trace: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], np.ndarray]:
    path = root / _relative(str(source["page_image"]))
    if not path.is_file() or _hash_file(path) != source["page_image_sha256"]:
        raise ThreeStockMtfError(f"RF3.D3 page image drift: {source['stock_id']}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or [image.shape[1], image.shape[0]] != source["page_image_size"]:
        raise ThreeStockMtfError(f"RF3.D3 page image size drift: {source['stock_id']}")

    x_anchors = [[math.log10(row[0]), row[1]] for row in source["x_log10_anchors"]]
    y_anchors = [[math.log10(row[0]), row[1]] for row in source["y_log10_anchors"]]
    grid = [int(value) for value in source["horizontal_grid_y"]]
    low_y, high_y = map(int, source["plot_y_bounds"])
    minimum_log = math.log10(float(trace["minimum_response_percent"]))
    maximum_log = math.log10(float(trace["maximum_response_percent"]))
    rows: list[dict[str, Any]] = []
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for frequency in (8.0, 15.0, 30.0, 40.0, 60.0):
        x = round(_linear_from_anchors(math.log10(frequency), x_anchors))
        candidates: list[tuple[int, float]] = []
        for y in range(low_y + 3, high_y - 3):
            if min(abs(y - line) for line in grid) <= int(
                trace["horizontal_grid_mask_radius_pixels"]
            ):
                continue
            hits = sum(
                int(image[y, xx] < int(trace["black_threshold_uint8"]))
                for xx in range(
                    x - int(trace["x_half_window_pixels"]),
                    x + int(trace["x_half_window_pixels"]) + 1,
                )
            )
            log_response = _value_from_pixel(float(y), y_anchors)
            if (
                hits >= int(trace["minimum_black_hits_per_row"])
                and minimum_log <= log_response <= maximum_log
            ):
                candidates.append((y, log_response))
        grouped = _clusters(candidates)
        centers = [
            (float(np.mean([item[0] for item in group])), float(np.mean([item[1] for item in group])))
            for group in grouped
        ]
        if not centers:
            raise ThreeStockMtfError(
                f"RF3.D3 no curve at {frequency}: {source['stock_id']}"
            )
        if int(source["expected_curve_count"]) == 1:
            selected = [min(centers, key=lambda item: item[0])]
        else:
            selected = [min(centers, key=lambda item: item[0]), max(centers, key=lambda item: item[0])]
        log_values = [item[1] for item in selected]
        midpoint = float(np.mean(log_values))
        for y, _ in selected:
            cv2.circle(overlay, (x, round(y)), 5, (255, 0, 0), 2)
        rows.append(
            {
                "frequency_cycles_per_mm": frequency,
                "x_pixel": x,
                "selected_y_pixels": [item[0] for item in selected],
                "envelope_response_percent": [10.0**max(log_values), 10.0**min(log_values)],
                "midpoint_log10_response": midpoint,
                "midpoint_response_percent": 10.0**midpoint,
            }
        )
    return {
        "stock_id": source["stock_id"],
        "published_curve_semantics": source["published_curve_semantics"],
        "rows": rows,
    }, overlay


def evaluate(
    contract: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    parent = root / _relative(str(contract["parent"]["path"]))
    if not parent.is_file() or _hash_file(parent) != contract["parent"]["sha256"]:
        raise ThreeStockMtfError("RF3.D3 parent evidence drift")
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    if parent_payload.get("decision") != contract["parent"]["required_decision"]:
        raise ThreeStockMtfError("RF3.D3 parent decision drift")

    traces: list[dict[str, Any]] = []
    overlay_hashes: dict[str, str] = {}
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for source in contract["sources"]:
        trace_result, overlay = _trace_source(source, contract["trace"], root)
        path = overlay_dir / f"{source['stock_id']}_overlay.png"
        Image.fromarray(overlay).save(path)
        overlay_hashes[source["stock_id"]] = _hash_file(path)
        traces.append(trace_result)

    comparisons: list[dict[str, Any]] = []
    med_gate = float(
        contract["gates"]["minimum_pairwise_median_absolute_log10_response_difference"]
    )
    rmse_gate = float(contract["gates"]["minimum_pairwise_rmse_log10_response_difference"])
    for left, right in itertools.combinations(traces, 2):
        left_values = np.asarray(
            [row["midpoint_log10_response"] for row in left["rows"]], dtype=np.float64
        )
        right_values = np.asarray(
            [row["midpoint_log10_response"] for row in right["rows"]], dtype=np.float64
        )
        difference = np.abs(left_values - right_values)
        median = float(np.median(difference))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        comparisons.append(
            {
                "left": left["stock_id"],
                "right": right["stock_id"],
                "median_absolute_log10_response_difference": median,
                "rmse_log10_response_difference": rmse,
                "median_gate_pass": median >= med_gate,
                "rmse_gate_pass": rmse >= rmse_gate,
            }
        )
    gate_results = {
        "all_frequencies_detected": all(len(row["rows"]) == 5 for row in traces),
        "source_semantics_reported": all(row["published_curve_semantics"] for row in traces),
        "all_pairwise_median_differences_material": all(
            row["median_gate_pass"] for row in comparisons
        ),
        "all_pairwise_rmse_differences_material": all(
            row["rmse_gate_pass"] for row in comparisons
        ),
    }
    automatic_pass = all(gate_results.values())
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "traces": traces,
        "pairwise_comparisons": comparisons,
        "overlay_sha256": overlay_hashes,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
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
