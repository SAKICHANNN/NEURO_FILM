"""U6.P2T first-party VISION3 characteristic-shape diversity audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "neuro_film.u6_p2t_kodak_vision3_characteristic_diversity_contract.v1"
TRACE_SCHEMA = "neuro_film.kodak_vision3_characteristic_curve_pixels.v1"
REPORT_SCHEMA = "neuro_film.u6_p2t_kodak_vision3_characteristic_diversity_report.v1"
CHANNELS = ("blue", "green", "red")
STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)


class CharacteristicDiversityError(RuntimeError):
    """Raised when a frozen source, trace or diversity contract drifts."""


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
        raise CharacteristicDiversityError("P2T paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    normalization = comparison.get("shape_normalization", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U6.P2T"
        or payload.get("trace", {}).get("sha256")
        != "e7d9d45cd56066d1cc770c3a506117fcddc4ae52417c6d69d4088e8d0da61dfb"
        or tuple(comparison.get("channels", ())) != CHANNELS
        or tuple(comparison.get("stocks", ())) != STOCKS
        or comparison.get("monotonic_repair")
        != "cumulative maximum in source density"
        or normalization.get("comparison_grid_min") != -0.1
        or normalization.get("comparison_grid_max") != 1.1
        or normalization.get("comparison_grid_samples") != 257
        or gates
        != {
            "required_stock_count": 3,
            "required_channel_count": 3,
            "minimum_points_per_channel": 12,
            "axis_max_residual_px": 1.0,
            "trace_max_ink_distance_px": 2.0,
            "maximum_monotonic_reversal_density": 0.01,
            "minimum_density_span": 1.5,
            "minimum_10_to_90_exposure_span": 2.0,
            "minimum_pair_shape_rmse_per_channel": 0.02,
            "minimum_material_channels_per_pair": 2,
            "minimum_material_pair_count": 2,
            "minimum_material_graph_degree_per_stock": 1,
            "maximum_digitization_uncertainty_fraction_of_shape_gate": 0.5,
            "digitization_uncertainty_px": 2.0,
            "two_byte_identical_audits": True,
            "visual_overlay_required": True,
        }
    ):
        raise CharacteristicDiversityError("P2T frozen contract drift")
    _relative_path(str(payload.get("trace", {}).get("path", "")))
    for value in payload.get("parent_evidence", {}).values():
        _relative_path(str(value))
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != TRACE_SCHEMA
        or tuple(payload.get("stocks", {})) != STOCKS
    ):
        raise CharacteristicDiversityError("P2T trace contract drift")
    for stock in STOCKS:
        row = payload["stocks"][stock]
        if tuple(row.get("curves", {})) != CHANNELS:
            raise CharacteristicDiversityError(f"P2T channel order drift: {stock}")
        for group in ("source_pdf", "source_graph"):
            _relative_path(str(row.get(group, {}).get("path", "")))
    return payload


def _value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if pixel0 == pixel1:
        raise CharacteristicDiversityError("invalid linear axis anchors")
    return float(value0 + (pixel - pixel0) / (pixel1 - pixel0) * (value1 - value0))


def _axis_residual(
    rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 == value1:
        raise CharacteristicDiversityError("invalid linear axis values")
    return float(
        max(
            (
                abs(
                    float(pixel)
                    - (
                        pixel0
                        + (float(value) - value0)
                        / (value1 - value0)
                        * (pixel1 - pixel0)
                    )
                )
                for value, pixel in rows
            ),
            default=0.0,
        )
    )


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


def _draw_overlay(graph_path: Path, row: Mapping[str, Any], output: Path) -> str:
    image = Image.open(graph_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"blue": (0, 92, 255), "green": (0, 180, 80), "red": (240, 40, 40)}
    for channel in CHANNELS:
        points = [tuple(map(int, point)) for point in row["curves"][channel]]
        draw.line(points, fill=colors[channel], width=2)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colors[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return _hash_file(output)


def _normalized_shape(
    exposure: np.ndarray, density: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    repaired = np.maximum.accumulate(density)
    density_span = float(repaired[-1] - repaired[0])
    if density_span <= 0.0:
        raise CharacteristicDiversityError("non-positive characteristic density span")
    normalized_density = (repaired - repaired[0]) / density_span
    x10 = float(np.interp(0.1, normalized_density, exposure))
    x90 = float(np.interp(0.9, normalized_density, exposure))
    if x90 <= x10:
        raise CharacteristicDiversityError("invalid 10-to-90 exposure span")
    normalized_exposure = (exposure - x10) / (x90 - x10)
    return normalized_exposure, normalized_density, density_span, x10, x90


def _trace_lineage_exact(root: Path, row: Mapping[str, Any]) -> bool:
    lineage = row.get("trace_lineage")
    if lineage is None:
        return True
    path = root / _relative_path(str(lineage["path"]))
    if not path.is_file() or _hash_file(path) != lineage["sha256"]:
        return False
    parent = json.loads(path.read_text(encoding="utf-8"))
    axis_keys = (
        "x_scale",
        "x_value_pixels",
        "x_tick_value_pixels",
        "y_scale",
        "y_value_pixels",
        "y_tick_value_pixels",
        "known_vertical_grid_pixels",
        "known_horizontal_grid_pixels",
    )
    return parent.get("curves") == row["curves"] and all(
        parent.get("graph_axes", {}).get(key) == row["graph_axes"].get(key)
        for key in axis_keys
    )


def audit_diversity(
    config: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    trace_path = root / _relative_path(str(config["trace"]["path"]))
    if not trace_path.is_file() or _hash_file(trace_path) != config["trace"]["sha256"]:
        raise CharacteristicDiversityError("P2T trace integrity mismatch")
    trace = load_trace(trace_path)
    gates = config["gates"]
    normalization = config["comparison"]["shape_normalization"]

    parent_integrity = True
    parent_hashes: dict[str, str] = {}
    for name, relative in config["parent_evidence"].items():
        path = root / _relative_path(str(relative))
        if not path.is_file():
            parent_integrity = False
            continue
        parent_hashes[name] = _hash_file(path)

    stocks: dict[str, Any] = {}
    normalized: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    source_integrity = True
    trace_integrity = True
    axis_gate = True
    ink_gate = True
    range_gate = True
    monotonic_gate = True
    uncertainty_gate = True
    overlay_hashes: dict[str, str] = {}

    for stock in STOCKS:
        row = trace["stocks"][stock]
        pdf = root / _relative_path(str(row["source_pdf"]["path"]))
        graph = root / _relative_path(str(row["source_graph"]["path"]))
        pdf_ok = (
            pdf.is_file()
            and pdf.stat().st_size == int(row["source_pdf"]["bytes"])
            and _hash_file(pdf) == row["source_pdf"]["sha256"]
        )
        graph_ok = (
            graph.is_file()
            and graph.stat().st_size == int(row["source_graph"]["bytes"])
            and _hash_file(graph) == row["source_graph"]["sha256"]
        )
        if not pdf_ok or not graph_ok:
            raise CharacteristicDiversityError(f"P2T source integrity mismatch: {stock}")
        source_integrity &= pdf_ok and graph_ok
        image = np.asarray(Image.open(graph).convert("L"))
        expected_shape = (
            int(row["source_graph"]["height"]),
            int(row["source_graph"]["width"]),
        )
        if image.shape != expected_shape:
            raise CharacteristicDiversityError(f"P2T graph dimensions mismatch: {stock}")

        axes = row["graph_axes"]
        residuals = {
            "x": _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"]),
            "y": _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"]),
        }
        axis_pass = max(residuals.values()) <= float(gates["axis_max_residual_px"])
        axis_gate &= axis_pass
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

        channel_rows: dict[str, Any] = {}
        normalized[stock] = {}
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
                raise CharacteristicDiversityError(f"P2T invalid trace: {stock}/{channel}")
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
            reversal = float(max(0.0, -np.min(np.diff(density))))
            x_norm, y_norm, density_span, x10, x90 = _normalized_shape(exposure, density)
            uncertainty_density = float(gates["digitization_uncertainty_px"]) * abs(
                (float(axes["y_value_pixels"][1][0]) - float(axes["y_value_pixels"][0][0]))
                / (float(axes["y_value_pixels"][1][1]) - float(axes["y_value_pixels"][0][1]))
            )
            uncertainty_normalized = uncertainty_density / density_span
            channel_range_pass = (
                density_span >= float(gates["minimum_density_span"])
                and x90 - x10 >= float(gates["minimum_10_to_90_exposure_span"])
            )
            channel_monotonic_pass = reversal <= float(
                gates["maximum_monotonic_reversal_density"]
            )
            channel_ink_pass = float(np.max(distances)) <= float(
                gates["trace_max_ink_distance_px"]
            )
            channel_uncertainty_pass = uncertainty_normalized <= (
                float(gates["minimum_pair_shape_rmse_per_channel"])
                * float(gates["maximum_digitization_uncertainty_fraction_of_shape_gate"])
            )
            range_gate &= channel_range_pass
            monotonic_gate &= channel_monotonic_pass
            ink_gate &= channel_ink_pass
            uncertainty_gate &= channel_uncertainty_pass
            normalized[stock][channel] = (x_norm, y_norm)
            channel_rows[channel] = {
                "coordinates_px": coordinates.tolist(),
                "log_relative_exposure": exposure.tolist(),
                "source_density": density.tolist(),
                "density_span": density_span,
                "maximum_monotonic_reversal_density": reversal,
                "trace_max_ink_distance_px": float(np.max(distances)),
                "x10_log_relative_exposure": x10,
                "x90_log_relative_exposure": x90,
                "x10_to_x90_span": x90 - x10,
                "digitization_uncertainty_normalized_density": uncertainty_normalized,
            }
        lineage_ok = _trace_lineage_exact(root, row)
        trace_integrity &= lineage_ok
        overlay = overlay_dir / f"{stock}.png"
        overlay_hashes[stock] = _draw_overlay(graph, row, overlay)
        stocks[stock] = {
            "source_pdf_sha256": row["source_pdf"]["sha256"],
            "source_graph_sha256": row["source_graph"]["sha256"],
            "measurement_context": row["measurement_context"],
            "axis_max_residual_px": residuals,
            "trace_lineage_exact": lineage_ok,
            "channels": channel_rows,
            "overlay_sha256": overlay_hashes[stock],
        }

    pairwise: dict[str, Any] = {}
    material_pairs: list[str] = []
    degrees = {stock: 0 for stock in STOCKS}
    for first_index, first in enumerate(STOCKS):
        for second in STOCKS[first_index + 1 :]:
            channels: dict[str, Any] = {}
            material_channel_count = 0
            for channel in CHANNELS:
                first_x, first_y = normalized[first][channel]
                second_x, second_y = normalized[second][channel]
                low = max(
                    float(first_x[0]),
                    float(second_x[0]),
                    float(normalization["comparison_grid_min"]),
                )
                high = min(
                    float(first_x[-1]),
                    float(second_x[-1]),
                    float(normalization["comparison_grid_max"]),
                )
                if high <= low:
                    raise CharacteristicDiversityError("empty normalized comparison domain")
                grid = np.linspace(
                    low, high, int(normalization["comparison_grid_samples"]), dtype=np.float64
                )
                first_shape = np.interp(grid, first_x, first_y)
                second_shape = np.interp(grid, second_x, second_y)
                shape_rmse = float(np.sqrt(np.mean(np.square(first_shape - second_shape))))

                first_source = stocks[first]["channels"][channel]
                second_source = stocks[second]["channels"][channel]
                direct_low = max(
                    float(first_source["log_relative_exposure"][0]),
                    float(second_source["log_relative_exposure"][0]),
                )
                direct_high = min(
                    float(first_source["log_relative_exposure"][-1]),
                    float(second_source["log_relative_exposure"][-1]),
                )
                direct_grid = np.linspace(direct_low, direct_high, 257, dtype=np.float64)
                direct_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.interp(
                                    direct_grid,
                                    np.asarray(first_source["log_relative_exposure"]),
                                    np.maximum.accumulate(np.asarray(first_source["source_density"])),
                                )
                                - np.interp(
                                    direct_grid,
                                    np.asarray(second_source["log_relative_exposure"]),
                                    np.maximum.accumulate(np.asarray(second_source["source_density"])),
                                )
                            )
                        )
                    )
                )
                material = shape_rmse >= float(gates["minimum_pair_shape_rmse_per_channel"])
                material_channel_count += int(material)
                channels[channel] = {
                    "normalized_shape_rmse": shape_rmse,
                    "direct_source_density_rmse": direct_rmse,
                    "normalized_comparison_domain": [low, high],
                    "material_shape_difference": material,
                }
            pair_id = f"{first}__vs__{second}"
            material_pair = material_channel_count >= int(
                gates["minimum_material_channels_per_pair"]
            )
            if material_pair:
                material_pairs.append(pair_id)
                degrees[first] += 1
                degrees[second] += 1
            pairwise[pair_id] = {
                "channels": channels,
                "material_channel_count": material_channel_count,
                "material_pair": material_pair,
            }

    diversity_gate = len(material_pairs) >= int(gates["minimum_material_pair_count"]) and all(
        degree >= int(gates["minimum_material_graph_degree_per_stock"])
        for degree in degrees.values()
    )
    gate_results = {
        "parent_evidence_present": parent_integrity,
        "source_integrity": source_integrity,
        "trace_integrity_and_lineage": trace_integrity,
        "axis_geometry": axis_gate,
        "source_ink_proximity": ink_gate,
        "source_domain_span": range_gate,
        "channel_monotonicity": monotonic_gate,
        "digitization_uncertainty_below_half_shape_gate": uncertainty_gate,
        "stock_shape_diversity": diversity_gate,
        "visual_overlays_generated": len(overlay_hashes) == len(STOCKS),
        "no_parameter_fit": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "trace_sha256": config["trace"]["sha256"],
        "parent_evidence_sha256": parent_hashes,
        "stocks": stocks,
        "pairwise": pairwise,
        "material_pairs": material_pairs,
        "material_graph_degree": degrees,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "diversity_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            config["decision_branches"]["pass"]
            if passed
            else config["decision_branches"]["fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "CHANNELS",
    "REPORT_SCHEMA",
    "SCHEMA",
    "STOCKS",
    "TRACE_SCHEMA",
    "CharacteristicDiversityError",
    "audit_diversity",
    "load_contract",
    "load_trace",
]
