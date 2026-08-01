"""U6.P2P first-party Kodak 250D characteristic-curve source audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "neuro_film.u6_p2p_kodak_250d_characteristic_source_contract.v1"
TRACE_SCHEMA = "neuro_film.kodak_250d_characteristic_curve_pixels.v1"
REPORT_SCHEMA = "neuro_film.u6_p2p_kodak_250d_characteristic_source_report.v1"
CHANNELS = ("blue", "green", "red")


class CharacteristicSourceError(RuntimeError):
    """Raised when the frozen P2P source or trace contract drifts."""


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


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CharacteristicSourceError("P2P paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    graph = payload.get("graph", {})
    trace = payload.get("trace", {})
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
            "x_quantity": "log relative exposure",
            "y_quantity": "density",
        }
        or graph.get("page_zero_based") != 2
        or graph.get("embedded_image_index") != 0
        or (graph.get("width"), graph.get("height")) != (587, 557)
        or graph.get("expected_bytes") != 121_845
        or graph.get("sha256")
        != "8d4ba7acec0be4a200ba4ab2f29f928bde63f430c8fd343d9a9e36bfc973ca0d"
        or trace.get("sha256")
        != "0157c8235160a47c231f091aa8fd42dae8a50a8af3c949cb17424a55b45394a9"
        or gates.get("axis_max_residual_px") != 1.0
        or gates.get("trace_max_ink_distance_px") != 2.0
        or gates.get("minimum_points_per_channel") != 12
        or gates.get("maximum_monotonic_reversal_density") != 0.01
        or not gates.get("require_blue_ge_green_ge_red")
        or not gates.get("two_byte_identical_audits")
        or not gates.get("visual_overlay_required")
    ):
        raise CharacteristicSourceError("P2P frozen contract drift")
    for value in (source.get("path", ""), graph.get("path", ""), trace.get("path", "")):
        _relative_path(str(value))
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    axes = payload.get("graph_axes", {})
    lineage = payload.get("annotation_lineage", {})
    if (
        payload.get("schema") != TRACE_SCHEMA
        or axes.get("x_scale") != "linear"
        or axes.get("y_scale") != "linear"
        or axes.get("x_value_pixels") != [[0.0, 62.0], [5.0, 506.0]]
        or axes.get("y_value_pixels") != [[0.0, 506.0], [3.0, 61.0]]
        or tuple(payload.get("curves", {})) != CHANNELS
        or lineage.get("subset") != "curves.250d_characteristic"
    ):
        raise CharacteristicSourceError("P2P trace contract drift")
    return payload


def _value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if pixel0 == pixel1:
        raise CharacteristicSourceError("invalid linear axis anchors")
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(value0 + fraction * (value1 - value0))


def _axis_residual(rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 == value1:
        raise CharacteristicSourceError("invalid linear axis values")
    residuals = [
        abs(
            float(pixel)
            - (
                pixel0
                + ((float(value) - value0) / (value1 - value0)) * (pixel1 - pixel0)
            )
        )
        for value, pixel in rows
    ]
    return float(max(residuals, default=0.0))


def _ink_distance(
    image: np.ndarray,
    x: int,
    y: int,
    maximum: float,
    *,
    excluded_x: set[int],
    excluded_y: set[int],
) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if xx in excluded_x or yy in excluded_y:
                continue
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
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colors[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return _hash_file(output)


def _lineage_exact(root: Path, trace: Mapping[str, Any]) -> bool:
    lineage = trace["annotation_lineage"]
    path = root / _relative_path(str(lineage["path"]))
    if not path.is_file() or _hash_file(path) != lineage["sha256"]:
        return False
    parent = json.loads(path.read_text(encoding="utf-8"))
    return parent.get("curves", {}).get("250d_characteristic") == trace["curves"]


def audit_source(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> dict[str, Any]:
    source_path = root / _relative_path(str(config["source"]["path"]))
    graph_path = root / _relative_path(str(config["graph"]["path"]))
    trace_path = root / _relative_path(str(config["trace"]["path"]))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(config["source"]["expected_bytes"])
        or _hash_file(source_path) != config["source"]["sha256"]
    ):
        raise CharacteristicSourceError("P2P source PDF integrity mismatch")
    if (
        not graph_path.is_file()
        or graph_path.stat().st_size != int(config["graph"]["expected_bytes"])
        or _hash_file(graph_path) != config["graph"]["sha256"]
    ):
        raise CharacteristicSourceError("P2P graph integrity mismatch")
    if not trace_path.is_file() or _hash_file(trace_path) != config["trace"]["sha256"]:
        raise CharacteristicSourceError("P2P trace integrity mismatch")

    image = np.asarray(Image.open(graph_path).convert("L"))
    if (image.shape[1], image.shape[0]) != (
        int(config["graph"]["width"]),
        int(config["graph"]["height"]),
    ):
        raise CharacteristicSourceError("P2P graph dimensions mismatch")
    trace = load_trace(trace_path)
    axes = trace["graph_axes"]
    gates = config["gates"]
    axis_residual = {
        "x": _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"]),
        "y": _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"]),
    }
    axis_gate = max(axis_residual.values()) <= float(gates["axis_max_residual_px"])
    excluded_x = {
        value
        for center in axes["known_vertical_grid_pixels"]
        for value in range(int(center) - 2, int(center) + 3)
    }
    excluded_y = {
        value
        for center in axes["known_horizontal_grid_pixels"]
        for value in range(int(center) - 2, int(center) + 3)
    }

    channels: dict[str, Any] = {}
    minimum = int(gates["minimum_points_per_channel"])
    for channel in CHANNELS:
        coordinates = np.asarray(trace["curves"][channel], dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < minimum:
            raise CharacteristicSourceError(f"P2P invalid trace point count: {channel}")
        if (
            np.any(coordinates[:, 0] < 0)
            or np.any(coordinates[:, 0] >= image.shape[1])
            or np.any(coordinates[:, 1] < 0)
            or np.any(coordinates[:, 1] >= image.shape[0])
        ):
            raise CharacteristicSourceError(f"P2P trace outside graph: {channel}")
        exposure = np.asarray(
            [_value_from_pixel(float(x), axes["x_value_pixels"]) for x in coordinates[:, 0]],
            dtype=np.float64,
        )
        density = np.asarray(
            [_value_from_pixel(float(y), axes["y_value_pixels"]) for y in coordinates[:, 1]],
            dtype=np.float64,
        )
        distances = np.asarray(
            [
                _ink_distance(
                    image,
                    int(x),
                    int(y),
                    float(gates["trace_max_ink_distance_px"]),
                    excluded_x=excluded_x,
                    excluded_y=excluded_y,
                )
                for x, y in coordinates
            ],
            dtype=np.float64,
        )
        channels[channel] = {
            "coordinates_px": coordinates.tolist(),
            "log_relative_exposure": exposure.tolist(),
            "status_m_density": density.tolist(),
            "trace_span_log_exposure": float(exposure[-1] - exposure[0]),
            "trace_max_ink_distance_px": float(np.max(distances)),
            "maximum_monotonic_reversal_density": float(max(0.0, -np.min(np.diff(density)))),
            "strictly_increasing_source_x": bool(np.all(np.diff(coordinates[:, 0]) > 0)),
        }

    exposure_min = max(row["log_relative_exposure"][0] for row in channels.values())
    exposure_max = min(row["log_relative_exposure"][-1] for row in channels.values())
    common_exposure = np.linspace(exposure_min, exposure_max, 257, dtype=np.float64)
    common_density = np.asarray(
        [
            np.interp(
                common_exposure,
                np.asarray(channels[channel]["log_relative_exposure"]),
                np.asarray(channels[channel]["status_m_density"]),
            )
            for channel in CHANNELS
        ],
        dtype=np.float64,
    )
    range_gate = all(
        min(row["log_relative_exposure"]) >= float(gates["log_exposure_min"])
        and max(row["log_relative_exposure"]) <= float(gates["log_exposure_max"])
        and min(row["status_m_density"]) >= float(gates["density_min"])
        and max(row["status_m_density"]) <= float(gates["density_max"])
        for row in channels.values()
    )
    span_gate = all(
        row["trace_span_log_exposure"] >= float(gates["minimum_trace_span_log_exposure"])
        for row in channels.values()
    )
    monotonic_gate = all(
        row["strictly_increasing_source_x"]
        and row["maximum_monotonic_reversal_density"]
        <= float(gates["maximum_monotonic_reversal_density"])
        for row in channels.values()
    )
    order_gate = bool(
        np.all(common_density[0] >= common_density[1])
        and np.all(common_density[1] >= common_density[2])
    )
    ink_gate = all(
        row["trace_max_ink_distance_px"] <= float(gates["trace_max_ink_distance_px"])
        for row in channels.values()
    )
    lineage_gate = _lineage_exact(root, trace)
    overlay_sha256 = _draw_overlay(graph_path, trace, overlay_path)
    stable = {
        "experiment_id": config["experiment_id"],
        "source_sha256": _hash_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "graph_sha256": _hash_file(graph_path),
        "trace_sha256": _hash_file(trace_path),
        "annotation_lineage_sha256": trace["annotation_lineage"]["sha256"],
        "overlay_sha256": overlay_sha256,
        "measurement_context": config["source"]["measurement_context"],
        "axis_max_residual_px": axis_residual,
        "common_order_domain_log_relative_exposure": [exposure_min, exposure_max],
        "channels": channels,
        "gate_results": {
            "axis_geometry": axis_gate,
            "trace_integrity_and_lineage": lineage_gate,
            "source_ink_proximity": ink_gate,
            "source_domain_range": range_gate,
            "source_domain_span": span_gate,
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
        "decision": "open_u6_p2q_characteristic_prior_compiler" if passed else "close_characteristic_trace",
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "TRACE_SCHEMA",
    "CharacteristicSourceError",
    "audit_source",
    "load_contract",
    "load_trace",
]
