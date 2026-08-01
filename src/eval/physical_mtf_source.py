"""U6.P5I first-party Kodak 250D MTF source and trace audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "neuro_film.u6_p5i_kodak_250d_mtf_source_contract.v1"
TRACE_SCHEMA = "neuro_film.kodak_250d_mtf_curve_pixels.v1"
REPORT_SCHEMA = "neuro_film.u6_p5i_kodak_250d_mtf_source_report.v1"
CHANNELS = ("blue", "green", "red")


class MtfSourceError(RuntimeError):
    """Raised when the frozen P5I source or trace contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MtfSourceError("P5I paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    graph = payload.get("graph", {})
    gates = payload.get("gates", {})
    context = source.get("measurement_context", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("expected_bytes") != 684_722
        or source.get("sha256")
        != "70adb298a7aabb285d986b720e07c87c27eb2361f0925aea2b15903c08282e16"
        or context
        != {
            "exposure": "5500 K daylight",
            "process": "ECN-2",
            "densitometry": "Status M",
            "spatial_frequency_unit": "cycles/mm",
            "response_unit": "percent",
        }
        or graph.get("page_zero_based") != 2
        or graph.get("embedded_image_index") != 2
        or (graph.get("width"), graph.get("height")) != (710, 589)
        or graph.get("sha256")
        != "c99ca3cefe1d028b68ecdbd8bae5f65fef601a4b8cefd6dc0907ca022beb4075"
        or gates.get("axis_max_residual_px") != 1.0
        or gates.get("frequency_trace_max_residual_cycles_per_mm") != 0.35
        or gates.get("trace_max_ink_distance_px") != 2.0
        or gates.get("minimum_points_per_channel") != 8
        or not gates.get("require_blue_ge_green_ge_red")
        or not gates.get("two_byte_identical_audits")
        or not gates.get("visual_overlay_required")
    ):
        raise MtfSourceError("P5I frozen contract drift")
    for value in (source.get("path", ""), graph.get("path", ""), payload.get("trace", "")):
        _relative_path(value)
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    axes = payload.get("graph_axes", {})
    curves = payload.get("curves", {})
    frequencies = payload.get("frequencies_cycles_per_mm", [])
    if (
        payload.get("schema") != TRACE_SCHEMA
        or axes.get("x_scale") != "log10"
        or axes.get("y_scale") != "log10"
        or axes.get("x_value_pixels") != [[1.0, 94.0], [600.0, 687.0]]
        or axes.get("y_value_pixels") != [[1.0, 520.0], [200.0, 72.0]]
        or tuple(curves) != CHANNELS
        or frequencies != [25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 60.0, 65.0]
        or any(len(curves.get(channel, [])) != len(frequencies) for channel in CHANNELS)
    ):
        raise MtfSourceError("P5I trace contract drift")
    return payload


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if pixel0 == pixel1 or value0 <= 0.0 or value1 <= 0.0:
        raise MtfSourceError("invalid logarithmic axis anchors")
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(10.0 ** (math.log10(value0) + fraction * (math.log10(value1) - math.log10(value0))))


def _axis_residual(rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    denominator = math.log10(value1) - math.log10(value0)
    residuals = []
    for value, pixel in rows:
        expected = pixel0 + (
            (math.log10(value) - math.log10(value0)) / denominator
        ) * (pixel1 - pixel0)
        residuals.append(abs(float(pixel) - expected))
    return float(max(residuals, default=0.0))


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] < 128:
                best = min(best, math.hypot(xx - x, yy - y))
    return float(best if math.isfinite(best) else maximum + 1.0)


def _draw_overlay(graph_path: Path, trace: Mapping[str, Any], output: Path) -> str:
    image = Image.open(graph_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"blue": (0, 92, 255), "green": (0, 180, 80), "red": (240, 40, 40)}
    for channel in CHANNELS:
        points = [tuple(map(int, point)) for point in trace["curves"][channel]]
        draw.line(points, fill=colors[channel], width=2)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=colors[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return _hash_file(output)


def audit_source(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> dict[str, Any]:
    source_path = root / _relative_path(str(config["source"]["path"]))
    graph_path = root / _relative_path(str(config["graph"]["path"]))
    trace_path = root / _relative_path(str(config["trace"]))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(config["source"]["expected_bytes"])
        or _hash_file(source_path) != config["source"]["sha256"]
    ):
        raise MtfSourceError("P5I source PDF integrity mismatch")
    if not graph_path.is_file() or _hash_file(graph_path) != config["graph"]["sha256"]:
        raise MtfSourceError("P5I graph integrity mismatch")
    image = np.asarray(Image.open(graph_path).convert("L"))
    if (image.shape[1], image.shape[0]) != (
        int(config["graph"]["width"]),
        int(config["graph"]["height"]),
    ):
        raise MtfSourceError("P5I graph dimensions mismatch")
    trace = load_trace(trace_path)
    axes = trace["graph_axes"]
    gates = config["gates"]
    axis_residual = {
        "x": _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"]),
        "y": _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"]),
    }
    axis_gate = max(axis_residual.values()) <= float(gates["axis_max_residual_px"])

    expected_frequencies = np.asarray(trace["frequencies_cycles_per_mm"], dtype=np.float64)
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        coordinates = np.asarray(trace["curves"][channel], dtype=np.int64)
        if coordinates.shape != (int(gates["minimum_points_per_channel"]), 2):
            raise MtfSourceError(f"P5I invalid trace point count: {channel}")
        if (
            np.any(coordinates[:, 0] < 0)
            or np.any(coordinates[:, 0] >= image.shape[1])
            or np.any(coordinates[:, 1] < 0)
            or np.any(coordinates[:, 1] >= image.shape[0])
        ):
            raise MtfSourceError(f"P5I trace outside graph: {channel}")
        frequencies = np.asarray(
            [_log_value_from_pixel(float(x), axes["x_value_pixels"]) for x in coordinates[:, 0]],
            dtype=np.float64,
        )
        responses = np.asarray(
            [_log_value_from_pixel(float(y), axes["y_value_pixels"]) for y in coordinates[:, 1]],
            dtype=np.float64,
        )
        ink_distances = np.asarray(
            [
                _ink_distance(image, int(x), int(y), float(gates["trace_max_ink_distance_px"]))
                for x, y in coordinates
            ],
            dtype=np.float64,
        )
        channels[channel] = {
            "coordinates_px": coordinates.tolist(),
            "frequencies_cycles_per_mm": frequencies.tolist(),
            "responses_percent": responses.tolist(),
            "frequency_max_residual_cycles_per_mm": float(
                np.max(np.abs(frequencies - expected_frequencies))
            ),
            "trace_max_ink_distance_px": float(np.max(ink_distances)),
            "monotonic_max_increase_percent": float(
                max(0.0, np.max(np.diff(responses), initial=-math.inf))
            ),
        }

    response_matrix = np.asarray(
        [channels[channel]["responses_percent"] for channel in CHANNELS],
        dtype=np.float64,
    )
    frequency_gate = all(
        row["frequency_max_residual_cycles_per_mm"]
        <= float(gates["frequency_trace_max_residual_cycles_per_mm"])
        for row in channels.values()
    )
    ink_gate = all(
        row["trace_max_ink_distance_px"] <= float(gates["trace_max_ink_distance_px"])
        for row in channels.values()
    )
    response_gate = bool(
        np.all(response_matrix >= float(gates["response_min_percent"]))
        and np.all(response_matrix <= float(gates["response_max_percent"]))
    )
    monotonic_gate = all(
        row["monotonic_max_increase_percent"]
        <= float(gates["maximum_monotonic_increase_percent"])
        for row in channels.values()
    )
    order_gate = bool(
        np.all(response_matrix[0] >= response_matrix[1])
        and np.all(response_matrix[1] >= response_matrix[2])
    )
    overlay_sha256 = _draw_overlay(graph_path, trace, overlay_path)
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": _hash_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "graph_sha256": _hash_file(graph_path),
        "trace_sha256": _hash_file(trace_path),
        "overlay_sha256": overlay_sha256,
        "measurement_context": config["source"]["measurement_context"],
        "axis_max_residual_px": axis_residual,
        "channels": channels,
        "gate_results": {
            "axis_geometry": axis_gate,
            "frequency_trace": frequency_gate,
            "source_ink_proximity": ink_gate,
            "response_range": response_gate,
            "channel_monotonicity": monotonic_gate,
            "blue_green_red_order": order_gate,
            "no_parameter_fit": True,
            "visual_overlay_generated": True,
        },
    }
    passed = all(stable["gate_results"].values())
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "source_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_u6_p5j_positive_psf_compiler" if passed else "close_mtf_trace",
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "TRACE_SCHEMA",
    "MtfSourceError",
    "audit_source",
    "load_contract",
    "load_trace",
]
