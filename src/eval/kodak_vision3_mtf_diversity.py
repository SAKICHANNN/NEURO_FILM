"""U5.R2BU0 first-party VISION3 source-domain MTF diversity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "neuro_film.u5_r2bu0_kodak_vision3_mtf_diversity_contract.v1"
TRACE_SCHEMA = "neuro_film.kodak_vision3_mtf_curve_pixels.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu0_kodak_vision3_mtf_diversity_report.v1"
STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
CHANNELS = ("blue", "green", "red")


class MtfDiversityError(RuntimeError):
    """Raised when a frozen source, trace or comparison contract drifts."""


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
        raise MtfDiversityError("BU0 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU0"
        or payload.get("trace", {}).get("sha256")
        != "d88c0e698c0e0969696224df50f1573ed78a6de0a894075f6502d9b3f658a794"
        or payload.get("parent_characteristic_decision", {}).get("sha256")
        != "9cd529029ad2495382750c6b61d3c0c167cdbb559ddc0cdecb0671411a14e6db"
        or tuple(comparison.get("stocks", ())) != STOCKS
        or tuple(comparison.get("channels", ())) != CHANNELS
        or comparison.get("common_frequencies_cycles_per_mm")
        != [25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 60.0, 65.0]
        or comparison.get("metric")
        != (
            "RMSE of natural-log MTF response after subtracting the per-channel "
            "value at 25 cycles/mm"
        )
        or comparison.get("interpolation")
        != "linear in log10 spatial frequency and natural-log response"
        or gates
        != {
            "required_stock_count": 3,
            "required_channel_count": 3,
            "minimum_points_per_channel": 8,
            "axis_max_residual_px": 1.0,
            "trace_max_ink_distance_px": 2.0,
            "minimum_response_percent": 10.0,
            "maximum_response_percent": 110.0,
            "maximum_monotonic_increase_percent": 2.0,
            "minimum_pair_log_shape_rmse_per_channel": 0.08,
            "minimum_material_channels_per_pair": 2,
            "minimum_material_pair_count": 3,
            "minimum_material_graph_degree_per_stock": 2,
            "digitization_uncertainty_px": 2.0,
            "maximum_digitization_uncertainty_fraction_of_shape_gate": 0.5,
            "two_byte_identical_audits": True,
            "visual_overlay_required": True,
        }
    ):
        raise MtfDiversityError("BU0 frozen contract drift")
    _relative_path(str(payload.get("trace", {}).get("path", "")))
    _relative_path(str(payload.get("parent_characteristic_decision", {}).get("path", "")))
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != TRACE_SCHEMA or tuple(payload.get("stocks", {})) != STOCKS:
        raise MtfDiversityError("BU0 trace contract drift")
    for stock in STOCKS:
        row = payload["stocks"][stock]
        if tuple(row.get("curves", {})) != CHANNELS:
            raise MtfDiversityError(f"BU0 channel order drift: {stock}")
        if row.get("graph_axes", {}).get("x_scale") != "log10" or row.get(
            "graph_axes", {}
        ).get("y_scale") != "log10":
            raise MtfDiversityError(f"BU0 axis scale drift: {stock}")
        for group in ("source_pdf", "source_graph"):
            _relative_path(str(row.get(group, {}).get("path", "")))
    return payload


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 <= 0.0 or value1 <= 0.0 or pixel0 == pixel1:
        raise MtfDiversityError("BU0 invalid logarithmic axis anchors")
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(
        10.0
        ** (math.log10(float(value0)) + fraction * (math.log10(float(value1)) - math.log10(float(value0))))
    )


def _axis_residual(
    rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    denominator = math.log10(float(value1)) - math.log10(float(value0))
    return float(
        max(
            (
                abs(
                    float(pixel)
                    - (
                        float(pixel0)
                        + (math.log10(float(value)) - math.log10(float(value0)))
                        / denominator
                        * (float(pixel1) - float(pixel0))
                    )
                )
                for value, pixel in rows
            ),
            default=0.0,
        )
    )


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] < 128:
                best = min(best, math.hypot(xx - x, yy - y))
    return float(best if math.isfinite(best) else maximum + 1.0)


def _draw_overlay(graph: Path, row: Mapping[str, Any], output: Path) -> str:
    image = Image.open(graph).convert("RGB")
    draw = ImageDraw.Draw(image)
    colours = {"blue": (0, 92, 255), "green": (0, 180, 80), "red": (240, 40, 40)}
    for channel in CHANNELS:
        points = [tuple(map(int, point)) for point in row["curves"][channel]]
        draw.line(points, fill=colours[channel], width=2)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colours[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return hash_file(output)


def _trace_lineage_exact(root: Path, row: Mapping[str, Any]) -> bool:
    lineage = row.get("trace_lineage")
    if lineage is None:
        return True
    path = root / _relative_path(str(lineage["path"]))
    if not path.is_file() or hash_file(path) != lineage["sha256"]:
        return False
    parent = json.loads(path.read_text(encoding="utf-8"))
    return parent.get("curves") == row["curves"] and parent.get("graph_axes") == row[
        "graph_axes"
    ]


def audit_diversity(
    config: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    trace_path = root / _relative_path(str(config["trace"]["path"]))
    if not trace_path.is_file() or hash_file(trace_path) != config["trace"]["sha256"]:
        raise MtfDiversityError("BU0 trace integrity mismatch")
    trace = load_trace(trace_path)
    parent_lock = config["parent_characteristic_decision"]
    parent_path = root / _relative_path(str(parent_lock["path"]))
    if not parent_path.is_file() or hash_file(parent_path) != parent_lock["sha256"]:
        raise MtfDiversityError("BU0 characteristic parent integrity mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_gate = bool(
        parent.get("decision") == parent_lock["required_decision"]
        and parent.get("result", {}).get("material_pair_count")
        == parent_lock["required_material_pair_count"]
    )

    gates = config["gates"]
    common = np.asarray(
        config["comparison"]["common_frequencies_cycles_per_mm"], dtype=np.float64
    )
    stocks: dict[str, Any] = {}
    normalized: dict[str, dict[str, np.ndarray]] = {}
    source_gate = True
    trace_gate = True
    axis_gate = True
    ink_gate = True
    response_gate = True
    monotonic_gate = True
    coverage_gate = True
    uncertainty_gate = True
    overlay_hashes: dict[str, str] = {}

    for stock in STOCKS:
        row = trace["stocks"][stock]
        pdf = root / _relative_path(str(row["source_pdf"]["path"]))
        graph = root / _relative_path(str(row["source_graph"]["path"]))
        pdf_ok = bool(
            pdf.is_file()
            and pdf.stat().st_size == int(row["source_pdf"]["bytes"])
            and hash_file(pdf) == row["source_pdf"]["sha256"]
        )
        graph_ok = bool(
            graph.is_file()
            and graph.stat().st_size == int(row["source_graph"]["bytes"])
            and hash_file(graph) == row["source_graph"]["sha256"]
        )
        if not pdf_ok or not graph_ok:
            raise MtfDiversityError(f"BU0 source integrity mismatch: {stock}")
        source_gate &= pdf_ok and graph_ok
        image = np.asarray(Image.open(graph).convert("L"))
        if image.shape != (
            int(row["source_graph"]["height"]),
            int(row["source_graph"]["width"]),
        ):
            raise MtfDiversityError(f"BU0 graph dimensions mismatch: {stock}")

        axes = row["graph_axes"]
        axis_residual = {
            "x": _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"]),
            "y": _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"]),
        }
        stock_axis_gate = max(axis_residual.values()) <= float(
            gates["axis_max_residual_px"]
        )
        axis_gate &= stock_axis_gate
        log_response_per_pixel = abs(
            math.log(float(axes["y_value_pixels"][1][0]))
            - math.log(float(axes["y_value_pixels"][0][0]))
        ) / abs(
            float(axes["y_value_pixels"][1][1])
            - float(axes["y_value_pixels"][0][1])
        )
        uncertainty = float(gates["digitization_uncertainty_px"]) * log_response_per_pixel
        stock_uncertainty_gate = uncertainty <= (
            float(gates["minimum_pair_log_shape_rmse_per_channel"])
            * float(gates["maximum_digitization_uncertainty_fraction_of_shape_gate"])
        )
        uncertainty_gate &= stock_uncertainty_gate

        normalized[stock] = {}
        channel_rows: dict[str, Any] = {}
        for channel in CHANNELS:
            coordinates = np.asarray(row["curves"][channel], dtype=np.int64)
            if (
                coordinates.ndim != 2
                or coordinates.shape[1] != 2
                or len(coordinates) < int(gates["minimum_points_per_channel"])
                or np.any(coordinates[:, 0] < 0)
                or np.any(coordinates[:, 0] >= image.shape[1])
                or np.any(coordinates[:, 1] < 0)
                or np.any(coordinates[:, 1] >= image.shape[0])
                or not np.all(np.diff(coordinates[:, 0]) > 0)
            ):
                raise MtfDiversityError(f"BU0 invalid trace: {stock}/{channel}")
            frequencies = np.asarray(
                [
                    _log_value_from_pixel(float(x), axes["x_value_pixels"])
                    for x in coordinates[:, 0]
                ],
                dtype=np.float64,
            )
            responses = np.asarray(
                [
                    _log_value_from_pixel(float(y), axes["y_value_pixels"])
                    for y in coordinates[:, 1]
                ],
                dtype=np.float64,
            )
            distances = np.asarray(
                [
                    _ink_distance(
                        image,
                        int(x),
                        int(y),
                        float(gates["trace_max_ink_distance_px"]),
                    )
                    for x, y in coordinates
                ],
                dtype=np.float64,
            )
            channel_coverage = bool(frequencies[0] <= common[0] and frequencies[-1] >= common[-1])
            channel_response = bool(
                np.all(responses >= float(gates["minimum_response_percent"]))
                and np.all(responses <= float(gates["maximum_response_percent"]))
            )
            max_increase = float(max(0.0, np.max(np.diff(responses), initial=-math.inf)))
            channel_monotonic = max_increase <= float(
                gates["maximum_monotonic_increase_percent"]
            )
            channel_ink = float(np.max(distances)) <= float(
                gates["trace_max_ink_distance_px"]
            )
            coverage_gate &= channel_coverage
            response_gate &= channel_response
            monotonic_gate &= channel_monotonic
            ink_gate &= channel_ink
            interpolated = np.interp(
                np.log10(common), np.log10(frequencies), np.log(responses)
            )
            normalized[stock][channel] = interpolated - interpolated[0]
            channel_rows[channel] = {
                "coordinates_px": coordinates.tolist(),
                "frequencies_cycles_per_mm": frequencies.tolist(),
                "responses_percent": responses.tolist(),
                "maximum_monotonic_increase_percent": max_increase,
                "trace_max_ink_distance_px": float(np.max(distances)),
                "common_log_shape": normalized[stock][channel].tolist(),
                "coverage_pass": channel_coverage,
            }
        lineage_exact = _trace_lineage_exact(root, row)
        trace_gate &= lineage_exact
        overlay_path = overlay_dir / f"{stock}.png"
        overlay_hashes[stock] = _draw_overlay(graph, row, overlay_path)
        stocks[stock] = {
            "source_pdf_sha256": hash_file(pdf),
            "source_graph_sha256": hash_file(graph),
            "measurement_context": row["measurement_context"],
            "axis_max_residual_px": axis_residual,
            "digitization_log_response_uncertainty": uncertainty,
            "trace_lineage_exact": lineage_exact,
            "channels": channel_rows,
        }

    pairwise: dict[str, Any] = {}
    degrees = {stock: 0 for stock in STOCKS}
    material_pair_count = 0
    threshold = float(gates["minimum_pair_log_shape_rmse_per_channel"])
    for left_index, left in enumerate(STOCKS):
        for right in STOCKS[left_index + 1 :]:
            channels: dict[str, Any] = {}
            material_channels = 0
            for channel in CHANNELS:
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(normalized[left][channel] - normalized[right][channel])
                        )
                    )
                )
                material = rmse >= threshold
                material_channels += int(material)
                channels[channel] = {
                    "normalized_log_shape_rmse": rmse,
                    "material": material,
                }
            pair_material = material_channels >= int(
                gates["minimum_material_channels_per_pair"]
            )
            if pair_material:
                material_pair_count += 1
                degrees[left] += 1
                degrees[right] += 1
            pairwise[f"{left}__vs__{right}"] = {
                "channels": channels,
                "material_channel_count": material_channels,
                "material_pair": pair_material,
            }

    diversity_gate = bool(
        material_pair_count >= int(gates["minimum_material_pair_count"])
        and all(
            degree >= int(gates["minimum_material_graph_degree_per_stock"])
            for degree in degrees.values()
        )
    )
    gate_results = {
        "source_integrity": source_gate,
        "trace_integrity": trace_gate,
        "parent_characteristic_remains_closed": parent_gate,
        "axis_geometry": axis_gate,
        "source_ink_proximity": ink_gate,
        "response_range": response_gate,
        "channel_monotonicity": monotonic_gate,
        "common_frequency_coverage": coverage_gate,
        "digitization_uncertainty": uncertainty_gate,
        "stock_mtf_diversity": diversity_gate,
    }
    diversity_pass = all(gate_results.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "trace_sha256": hash_file(trace_path),
        "parent_characteristic_decision_sha256": hash_file(parent_path),
        "common_frequencies_cycles_per_mm": common.tolist(),
        "stocks": stocks,
        "pairwise": pairwise,
        "material_pair_count": material_pair_count,
        "material_graph_degree": degrees,
        "overlay_sha256": overlay_hashes,
        "gate_results": gate_results,
        "diversity_pass": diversity_pass,
        "decision": (
            "retain_source_domain_mtf_signature_and_open_joint_physical_prior"
            if diversity_pass
            else "close_exact_vision3_mtf_diversity_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable["stable_evidence_id"] = hashlib.sha256(canonical_json(stable)).hexdigest()
    return stable


__all__ = [
    "MtfDiversityError",
    "audit_diversity",
    "canonical_json",
    "hash_file",
    "load_contract",
    "load_trace",
]
