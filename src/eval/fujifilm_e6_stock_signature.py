"""U5.R2CB0 same-manufacturer Fujifilm E-6 source-signature audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader

SCHEMA = "neuro_film.u5_r2cb0_fujifilm_e6_stock_source_signature_contract.v1"
TRACE_SCHEMA = "neuro_film.fujifilm_e6_stock_mtf_trace.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb0_fujifilm_e6_stock_source_signature_report.v1"
EXPERIMENT_ID = "U5.R2CB0"
VELVIA50 = "fujichrome_velvia_50_rvp50_af3_0221e2"
VELVIA100 = "fujichrome_velvia_100_rvp100_af3_202e"
PROVIA100F = "fujichrome_provia_100f_rdpiii_af3_036e"
STOCKS = (VELVIA50, VELVIA100, PROVIA100F)
MTF_STOCKS = (VELVIA100, PROVIA100F)
CONTRACT_SHA256 = "f3351cf46d6159564de7761f04834395bcf5c8fc9e93e98e53d39d3638e7e008"


class FujifilmE6SignatureError(RuntimeError):
    """Raised when a frozen CB0 input or contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmE6SignatureError("CB0 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmE6SignatureError("CB0 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or tuple(payload.get("stocks", ())) != STOCKS
        or tuple(payload.get("comparison", {}).get("mtf_stocks", ())) != MTF_STOCKS
        or payload.get("comparison", {}).get("common_frequencies_cycles_per_mm")
        != [10.0, 20.0, 30.0, 40.0, 50.0, 59.0]
        or payload.get("velvia50_parent_trace", {}).get("allowed_observation")
        != "diffuse_rms_granularity_and_measurement_context_only"
        or payload.get("velvia50_parent_trace", {}).get("forbidden_observation")
        != "mtf_curve_or_axis_calibration"
    ):
        raise FujifilmE6SignatureError("CB0 frozen contract drift")
    for key in ("trace", "velvia50_parent_trace", "velvia50_parent_decision"):
        _relative_path(str(payload[key]["path"]))
    return payload


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 <= 0.0 or value1 <= 0.0 or pixel0 == pixel1:
        raise FujifilmE6SignatureError("CB0 invalid logarithmic axis anchors")
    fraction = (float(pixel) - float(pixel0)) / (float(pixel1) - float(pixel0))
    return float(
        math.exp(
            math.log(float(value0))
            + fraction * (math.log(float(value1)) - math.log(float(value0)))
        )
    )


def _axis_residual(
    ticks: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    denominator = math.log(float(value1)) - math.log(float(value0))
    return float(
        max(
            abs(
                float(pixel)
                - (
                    float(pixel0)
                    + (math.log(float(value)) - math.log(float(value0)))
                    / denominator
                    * (float(pixel1) - float(pixel0))
                )
            )
            for value, pixel in ticks
        )
    )


def _curve_values(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(row["curve"], dtype=np.float64)
    axes = row["graph_axes"]
    frequencies = np.asarray(
        [_log_value_from_pixel(x, axes["x_value_pixels"]) for x in points[:, 0]]
    )
    responses = np.asarray(
        [_log_value_from_pixel(y, axes["y_value_pixels"]) for y in points[:, 1]]
    )
    if not np.all(np.diff(frequencies) > 0.0):
        raise FujifilmE6SignatureError("CB0 trace frequencies must increase")
    return frequencies, responses


def _normalized_log_curve(
    frequencies: np.ndarray, responses: np.ndarray, common: np.ndarray
) -> np.ndarray:
    if common[0] < frequencies[0] or common[-1] > frequencies[-1]:
        raise FujifilmE6SignatureError("CB0 common frequencies exceed trace support")
    values = np.interp(np.log(common), np.log(frequencies), np.log(responses))
    return values - values[0]


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] < 128:
                best = min(best, math.hypot(xx - x, yy - y))
    return float(best if math.isfinite(best) else maximum + 1.0)


def _draw_overlay(graph: Path, points: Sequence[Sequence[int]], output: Path) -> str:
    image = Image.open(graph).convert("RGB")
    draw = ImageDraw.Draw(image)
    xy = [tuple(map(int, point)) for point in points]
    draw.line(xy, fill=(255, 48, 48), width=2)
    for x, y in xy:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 220, 255), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return hash_file(output)


def _verify_embedded_graph(pdf: Path, graph: Path, page: int, index: int) -> bool:
    embedded = PdfReader(pdf).pages[page - 1].images[index].image.convert("1")
    extracted = Image.open(graph).convert("1")
    return bool(
        embedded.size == extracted.size
        and (
            np.array_equal(np.asarray(embedded), np.asarray(extracted))
            or np.array_equal(np.asarray(embedded), ~np.asarray(extracted))
        )
    )


def _verify_graph_derivation(root: Path, row: Mapping[str, Any], graph: Path) -> bool:
    graph_info = row["source_graph"]
    if "embedded_page_image_index_zero_based" in graph_info:
        pdf = root / _relative_path(row["source_pdf"]["path"])
        return _verify_embedded_graph(
            pdf,
            graph,
            int(graph_info["page_one_based"]),
            int(graph_info["embedded_page_image_index_zero_based"]),
        )
    derivation = graph_info["derivation"]
    full = root / _relative_path(derivation["full_page_render_path"])
    if (
        not full.is_file()
        or hash_file(full) != derivation["full_page_render_sha256"]
        or Image.open(full).size
        != (
            int(derivation["full_page_render_width"]),
            int(derivation["full_page_render_height"]),
        )
    ):
        return False
    crop = Image.open(full).crop(tuple(map(int, derivation["crop_box_left_top_right_bottom"])))
    return bool(np.array_equal(np.asarray(crop), np.asarray(Image.open(graph))))


def _stable_identity(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("report_sha256", None)
    stable.pop("stable_evidence_id", None)
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def audit_signature(
    config: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    gates = config["gates"]
    inputs: dict[str, Any] = {}
    for key in ("trace", "velvia50_parent_trace", "velvia50_parent_decision"):
        path = root / _relative_path(config[key]["path"])
        expected = str(config[key]["sha256"])
        if not path.is_file() or hash_file(path) != expected:
            raise FujifilmE6SignatureError(f"CB0 input integrity mismatch: {key}")
        inputs[key] = json.loads(path.read_text(encoding="utf-8"))

    trace = inputs["trace"]
    parent = inputs["velvia50_parent_trace"]
    parent_decision = inputs["velvia50_parent_decision"]
    if trace.get("schema") != TRACE_SCHEMA or tuple(trace.get("stocks", {})) != MTF_STOCKS:
        raise FujifilmE6SignatureError("CB0 trace structure drift")
    if config["velvia50_parent_decision"]["required_failed_gate"] not in parent_decision.get(
        "failed_gates", []
    ):
        raise FujifilmE6SignatureError("CB0 parent invalid-axis boundary missing")

    parent_row = parent["revisions"][VELVIA50]
    parent_context = parent_row["measurement_context"]
    contexts = {VELVIA50: parent_context}
    granularity = {VELVIA50: float(parent_context["diffuse_rms_granularity"])}
    common = np.asarray(config["comparison"]["common_frequencies_cycles_per_mm"])
    normalized: dict[str, np.ndarray] = {}
    stocks: dict[str, Any] = {}
    source_gate = derivation_gate = axis_gate = ink_gate = response_gate = True

    for stock in MTF_STOCKS:
        row = trace["stocks"][stock]
        pdf = root / _relative_path(row["source_pdf"]["path"])
        graph = root / _relative_path(row["source_graph"]["path"])
        source_ok = bool(
            pdf.is_file()
            and pdf.stat().st_size == int(row["source_pdf"]["bytes"])
            and hash_file(pdf) == row["source_pdf"]["sha256"]
            and graph.is_file()
            and graph.stat().st_size == int(row["source_graph"]["bytes"])
            and hash_file(graph) == row["source_graph"]["sha256"]
            and Image.open(graph).size
            == (int(row["source_graph"]["width"]), int(row["source_graph"]["height"]))
        )
        if not source_ok:
            raise FujifilmE6SignatureError(f"CB0 source integrity mismatch: {stock}")
        source_gate &= source_ok
        derivation_ok = _verify_graph_derivation(root, row, graph)
        derivation_gate &= derivation_ok
        axes = row["graph_axes"]
        x_residual = _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"])
        y_residual = _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"])
        axis_gate &= max(x_residual, y_residual) <= float(gates["axis_max_residual_px"])
        image = np.asarray(Image.open(graph).convert("L"))
        ink_distances = [
            _ink_distance(image, int(x), int(y), float(gates["trace_max_ink_distance_px"]))
            for x, y in row["curve"]
        ]
        ink_gate &= max(ink_distances) <= float(gates["trace_max_ink_distance_px"])
        frequencies, responses = _curve_values(row)
        response_ok = bool(
            responses.min() >= float(gates["minimum_response_percent"])
            and responses.max() <= float(gates["maximum_response_percent"])
        )
        response_gate &= response_ok
        normalized[stock] = _normalized_log_curve(frequencies, responses, common)
        contexts[stock] = row["measurement_context"]
        granularity[stock] = float(row["measurement_context"]["diffuse_rms_granularity"])
        overlay_path = overlay_dir / f"{stock}_trace_overlay.png"
        stocks[stock] = {
            "axis_max_residual_px": max(x_residual, y_residual),
            "curve_frequencies_cycles_per_mm": frequencies.tolist(),
            "curve_responses_percent": responses.tolist(),
            "derivation_verified": derivation_ok,
            "ink_max_distance_px": max(ink_distances),
            "normalized_log_response": normalized[stock].tolist(),
            "overlay_sha256": _draw_overlay(graph, row["curve"], overlay_path),
            "source_verified": source_ok,
        }

    context_gate = all(
        context["process"] == gates["required_process"]
        and context["exposure"] == gates["required_exposure"]
        and int(context["granularity_aperture_micrometres"])
        == int(gates["required_granularity_aperture_micrometres"])
        and float(context["granularity_sample_density_above_minimum"])
        == float(gates["required_granularity_sample_density_above_minimum"])
        for context in contexts.values()
    )
    mtf_rmse = float(np.sqrt(np.mean((normalized[VELVIA100] - normalized[PROVIA100F]) ** 2)))
    slopes = {}
    for stock in MTF_STOCKS:
        axes = trace["stocks"][stock]["graph_axes"]["y_value_pixels"]
        slopes[stock] = abs(
            math.log(float(axes[1][0]) / float(axes[0][0]))
            / (float(axes[1][1]) - float(axes[0][1]))
        )
    uncertainty_bound = 2.0 * float(gates["digitization_uncertainty_px"]) * sum(
        slopes.values()
    )
    uncertainty_fraction = uncertainty_bound / mtf_rmse if mtf_rmse > 0.0 else math.inf

    pairs: dict[str, Any] = {}
    material_count = 0
    for index, first in enumerate(STOCKS):
        for second in STOCKS[index + 1 :]:
            pair_name = f"{first}__{second}"
            pair_mtf = mtf_rmse if {first, second} == set(MTF_STOCKS) else None
            granularity_difference = abs(granularity[first] - granularity[second])
            material = bool(
                (pair_mtf is not None and pair_mtf >= gates["minimum_required_pair_mtf_log_shape_rmse"])
                or granularity_difference >= gates["minimum_granularity_absolute_difference"]
            )
            material_count += int(material)
            pairs[pair_name] = {
                "granularity_absolute_difference": granularity_difference,
                "material": material,
                "mtf_log_shape_rmse": pair_mtf,
            }

    gate_results = {
        "axis_calibration": axis_gate,
        "graph_derivation": derivation_gate,
        "like_for_like_measurement_context": context_gate,
        "material_all_stock_pairs": material_count == gates["required_material_pair_count"],
        "required_velvia100_provia100f_mtf_separation": mtf_rmse
        >= gates["minimum_required_pair_mtf_log_shape_rmse"],
        "response_range": response_gate,
        "source_integrity": source_gate,
        "trace_ink": ink_gate,
        "trace_point_count": all(
            len(trace["stocks"][stock]["curve"])
            >= gates["minimum_trace_points_per_new_stock"]
            for stock in MTF_STOCKS
        ),
        "uncertainty_fraction": uncertainty_fraction
        <= gates["maximum_uncertainty_fraction_of_observed_mtf_separation"],
        "velvia50_invalid_mtf_excluded": True,
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "claim_ceiling": config["claim_ceiling"],
        "common_frequencies_cycles_per_mm": common.tolist(),
        "decision": "retain_non_renderable_fujifilm_e6_stock_source_signature"
        if passed
        else "close_exact_fujifilm_e6_stock_source_signature",
        "failed_gates": [key for key, value in gate_results.items() if not value],
        "gate_results": gate_results,
        "granularity": granularity,
        "material_pair_count": material_count,
        "mtf_digitization_uncertainty_bound": uncertainty_bound,
        "mtf_digitization_uncertainty_fraction": uncertainty_fraction,
        "mtf_log_shape_rmse": mtf_rmse,
        "pairs": pairs,
        "passed": passed,
        "stocks": stocks,
    }
    report["stable_evidence_id"] = _stable_identity(report)
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
