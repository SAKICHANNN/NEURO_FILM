"""U6.P4AV first-party Kodak 250D diffuse-rms granularity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "neuro_film.u6_p4av_kodak_250d_granularity_source_contract.v1"
TRACE_SCHEMA = "neuro_film.kodak_250d_granularity_curve_pixels.v1"
REPORT_SCHEMA = "neuro_film.u6_p4av_kodak_250d_granularity_source_report.v1"
CHANNELS = ("blue", "green", "red")


class GranularitySourceError(RuntimeError):
    """Raised when the frozen P4AV source or trace drifts."""


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


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise GranularitySourceError("P4AV paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    graph = payload.get("graph", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("expected_bytes") != 684_722
        or source.get("sha256")
        != "70adb298a7aabb285d986b720e07c87c27eb2361f0925aea2b15903c08282e16"
        or source.get("measurement_context")
        != {
            "exposure": "5500 K daylight",
            "process": "ECN-2",
            "densitometry": "Status M",
            "aperture_micrometres": 48.0,
            "quantity": "diffuse rms granularity Sigma D",
        }
        or (graph.get("width"), graph.get("height")) != (587, 557)
        or graph.get("expected_bytes") != 121_845
        or graph.get("sha256")
        != "8d4ba7acec0be4a200ba4ab2f29f928bde63f430c8fd343d9a9e36bfc973ca0d"
        or gates.get("axis_max_residual_px") != 1.5
        or gates.get("trace_max_ink_distance_px") != 2.0
        or gates.get("minimum_points_per_channel") != 18
        or not gates.get("two_byte_identical_audits")
        or not gates.get("visual_overlay_required")
    ):
        raise GranularitySourceError("P4AV frozen contract drift")
    for value in (source.get("path", ""), graph.get("path", ""), payload.get("trace", "")):
        _relative(str(value))
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    axes = payload.get("graph_axes", {})
    if (
        payload.get("schema") != TRACE_SCHEMA
        or axes.get("x_scale") != "linear"
        or axes.get("y_scale") != "log10"
        or axes.get("x_value_pixels") != [[0.0, 62.0], [5.0, 506.0]]
        or axes.get("y_value_pixels") != [[0.001, 506.0], [1.0, 61.0]]
        or tuple(payload.get("curves", {})) != CHANNELS
    ):
        raise GranularitySourceError("P4AV trace contract drift")
    return payload


def _linear_value(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (v0, p0), (v1, p1) = anchors
    if p0 == p1:
        raise GranularitySourceError("invalid linear axis")
    return float(v0 + ((pixel - p0) / (p1 - p0)) * (v1 - v0))


def _log_value(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (v0, p0), (v1, p1) = anchors
    if p0 == p1 or v0 <= 0.0 or v1 <= 0.0:
        raise GranularitySourceError("invalid log axis")
    fraction = (pixel - p0) / (p1 - p0)
    return float(10.0 ** (math.log10(v0) + fraction * (math.log10(v1) - math.log10(v0))))


def _axis_residual(
    rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]], *, log: bool
) -> float:
    (v0, p0), (v1, p1) = anchors
    residuals: list[float] = []
    for value, pixel in rows:
        if log:
            fraction = (math.log10(value) - math.log10(v0)) / (
                math.log10(v1) - math.log10(v0)
            )
        else:
            fraction = (value - v0) / (v1 - v0)
        residuals.append(abs(float(pixel) - (p0 + fraction * (p1 - p0))))
    return float(max(residuals, default=0.0))


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] < 128:
                best = min(best, math.hypot(xx - x, yy - y))
    return float(best if math.isfinite(best) else maximum + 1.0)


def _draw_overlay(graph: Path, trace: Mapping[str, Any], output: Path) -> str:
    image = Image.open(graph).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"blue": (0, 92, 255), "green": (0, 180, 80), "red": (240, 40, 40)}
    for channel in CHANNELS:
        points = [tuple(map(int, row)) for row in trace["curves"][channel]]
        draw.line(points, fill=colors[channel], width=2)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colors[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return _hash_file(output)


def audit_source(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> dict[str, Any]:
    source = root / _relative(str(config["source"]["path"]))
    graph = root / _relative(str(config["graph"]["path"]))
    trace_path = root / _relative(str(config["trace"]))
    if (
        not source.is_file()
        or source.stat().st_size != int(config["source"]["expected_bytes"])
        or _hash_file(source) != config["source"]["sha256"]
    ):
        raise GranularitySourceError("P4AV source PDF integrity mismatch")
    if (
        not graph.is_file()
        or graph.stat().st_size != int(config["graph"]["expected_bytes"])
        or _hash_file(graph) != config["graph"]["sha256"]
    ):
        raise GranularitySourceError("P4AV graph integrity mismatch")
    if not trace_path.is_file() or _hash_file(trace_path) != config["trace_sha256"]:
        raise GranularitySourceError("P4AV trace integrity mismatch")

    trace = load_trace(trace_path)
    image = np.asarray(Image.open(graph).convert("L"))
    if (image.shape[1], image.shape[0]) != (
        int(config["graph"]["width"]),
        int(config["graph"]["height"]),
    ):
        raise GranularitySourceError("P4AV graph dimensions mismatch")
    axes = trace["graph_axes"]
    gates = config["gates"]
    axis_residual = {
        "x": _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"], log=False),
        "y": _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"], log=True),
    }
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        coordinates = np.asarray(trace["curves"][channel], dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < int(
            gates["minimum_points_per_channel"]
        ):
            raise GranularitySourceError(f"P4AV invalid trace count: {channel}")
        if not np.all(np.diff(coordinates[:, 0]) > 0):
            raise GranularitySourceError(f"P4AV non-increasing trace x: {channel}")
        distances = np.asarray(
            [_ink_distance(image, int(x), int(y), float(gates["trace_max_ink_distance_px"])) for x, y in coordinates],
            dtype=np.float64,
        )
        exposure = np.asarray(
            [_linear_value(float(x), axes["x_value_pixels"]) for x in coordinates[:, 0]],
            dtype=np.float64,
        )
        sigma_d = np.asarray(
            [_log_value(float(y), axes["y_value_pixels"]) for y in coordinates[:, 1]],
            dtype=np.float64,
        )
        channels[channel] = {
            "coordinates_px": coordinates.tolist(),
            "log_relative_exposure": exposure.tolist(),
            "sigma_d": sigma_d.tolist(),
            "trace_max_ink_distance_px": float(np.max(distances)),
            "trace_span_log_exposure": float(exposure[-1] - exposure[0]),
            "sigma_d_min": float(np.min(sigma_d)),
            "sigma_d_max": float(np.max(sigma_d)),
        }

    overlay_sha = _draw_overlay(graph, trace, overlay_path)
    gate_results = {
        "axis_geometry": max(axis_residual.values()) <= float(gates["axis_max_residual_px"]),
        "source_ink_proximity": all(
            row["trace_max_ink_distance_px"] <= float(gates["trace_max_ink_distance_px"])
            for row in channels.values()
        ),
        "trace_span": all(
            row["trace_span_log_exposure"] >= float(gates["minimum_trace_span_log_exposure"])
            for row in channels.values()
        ),
        "physical_range": all(
            row["sigma_d_min"] >= float(gates["minimum_sigma_d"])
            and row["sigma_d_max"] <= float(gates["maximum_sigma_d"])
            for row in channels.values()
        ),
        "no_parameter_fit": True,
        "visual_overlay_generated": True,
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": _hash_file(source),
        "graph_sha256": _hash_file(graph),
        "trace_sha256": _hash_file(trace_path),
        "overlay_sha256": overlay_sha,
        "measurement_context": config["source"]["measurement_context"],
        "axis_max_residual_px": axis_residual,
        "channels": channels,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "source_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_joint_mtf_granularity_compatibility" if passed else "close_granularity_trace",
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "TRACE_SCHEMA",
    "GranularitySourceError",
    "audit_source",
    "load_contract",
    "load_trace",
]
