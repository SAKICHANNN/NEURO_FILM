"""U5.R2BU8 exact-vector VERITA 200D source-signature evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader
from pypdf.generic import ContentStream

SCHEMA = "neuro_film.u5_r2bu8_kodak_verita_200d_source_signature_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu8_kodak_verita_200d_source_signature_report.v1"
STOCK = "kodak_verita_200d_5206_7206"
VISION3_STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
CHANNELS = ("blue", "green", "red")
DOMAINS = ("characteristic", "mtf", "granularity")


class VeritaSourceSignatureError(RuntimeError):
    """Raised when a frozen source, vector path, or parent evidence drifts."""


@dataclass(frozen=True)
class Segment:
    kind: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class StrokedPath:
    operation_index: int
    width: float
    dash: tuple[float, ...]
    segments: tuple[Segment, ...]


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
        raise VeritaSourceSignatureError("BU8 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector = payload.get("vector_source", {})
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU8"
        or payload.get("source", {}).get("stock_id") != STOCK
        or payload.get("source", {}).get("sha256")
        != "aa5bfd899003cf4a0ae5a86a33325e0eac2ed16b9fca61c8894532de1458bae2"
        or vector.get("page_zero_based") != 2
        or vector.get("decoded_content_bytes") != 48293
        or vector.get("decoded_content_sha256")
        != "2f67a6759ca80d4839830c4037c00e268e253f0bb32c863d2f191e9e18f109f9"
        or tuple(comparison.get("vision3_stocks", ())) != VISION3_STOCKS
        or tuple(comparison.get("channels", ())) != CHANNELS
        or gates
        != {
            "required_source_pages": 5,
            "required_vector_domains": 3,
            "required_channels_per_domain": 3,
            "minimum_material_channels_per_domain_pair": 2,
            "minimum_material_domains_per_vision3_pair": 2,
            "minimum_material_vision3_pairs": 3,
            "maximum_characteristic_monotonic_reversal_density": 0.01,
            "minimum_characteristic_density_span": 1.5,
            "minimum_characteristic_10_to_90_exposure_span": 2.0,
            "minimum_mtf_response_percent": 1.0,
            "maximum_mtf_response_percent": 110.0,
            "maximum_mtf_monotonic_increase_percent": 2.0,
            "minimum_granularity_sigma_d": 0.001,
            "maximum_granularity_sigma_d": 0.05,
            "two_byte_identical_audits": True,
            "visual_overlay_required": True,
        }
    ):
        raise VeritaSourceSignatureError("BU8 frozen contract drift")
    for group in ("source",):
        _relative_path(str(payload[group]["path"]))
    for row in payload.get("vision3_evidence", {}).values():
        _relative_path(str(row.get("path", "")))
    for domain in DOMAINS:
        row = vector.get(domain, {})
        if tuple(row.get("channels", {})) != CHANNELS:
            raise VeritaSourceSignatureError(f"BU8 {domain} channel order drift")
    return payload


def _matrix_multiply(
    first: tuple[float, ...], second: tuple[float, ...]
) -> tuple[float, ...]:
    a, b, c, d, e, f = first
    g, h, i, j, k, l = second
    return (
        a * g + b * i,
        a * h + b * j,
        c * g + d * i,
        c * h + d * j,
        e * g + f * i + k,
        e * h + f * j + l,
    )


def _transform(
    matrix: tuple[float, ...], point: Sequence[float]
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = map(float, point)
    return (a * x + c * y + e, b * x + d * y + f)


def _dash_values(operands: Sequence[Any]) -> tuple[float, ...]:
    if not operands:
        return ()
    values = operands[0]
    return tuple(float(value) for value in values)


def _extract_stroked_paths(page: Any) -> tuple[dict[int, StrokedPath], bytes]:
    raw = page.get_contents().get_data()
    operations = ContentStream(page.get_contents(), page.pdf).operations
    state_stack: list[tuple[tuple[float, ...], float, tuple[float, ...]]] = []
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    width = 1.0
    dash: tuple[float, ...] = ()
    current: tuple[float, float] | None = None
    segments: list[Segment] = []
    paths: dict[int, StrokedPath] = {}
    for index, (operands, operator_raw) in enumerate(operations):
        operator = (
            operator_raw.decode("latin-1")
            if isinstance(operator_raw, bytes)
            else str(operator_raw)
        )
        if operator == "q":
            state_stack.append((matrix, width, dash))
        elif operator == "Q":
            if not state_stack:
                raise VeritaSourceSignatureError("BU8 unbalanced PDF graphics state")
            matrix, width, dash = state_stack.pop()
        elif operator == "cm":
            matrix = _matrix_multiply(matrix, tuple(float(value) for value in operands))
        elif operator == "w":
            width = float(operands[0])
        elif operator == "d":
            dash = _dash_values(operands)
        elif operator == "m":
            current = _transform(matrix, operands)
            segments = []
        elif operator == "l":
            if current is None:
                raise VeritaSourceSignatureError("BU8 line path has no start")
            endpoint = _transform(matrix, operands)
            segments.append(Segment("line", (current, endpoint)))
            current = endpoint
        elif operator == "c":
            if current is None:
                raise VeritaSourceSignatureError("BU8 cubic path has no start")
            control1 = _transform(matrix, operands[0:2])
            control2 = _transform(matrix, operands[2:4])
            endpoint = _transform(matrix, operands[4:6])
            segments.append(Segment("cubic", (current, control1, control2, endpoint)))
            current = endpoint
        elif operator in ("S", "s"):
            paths[index] = StrokedPath(index, width, dash, tuple(segments))
            current = None
            segments = []
        elif operator in ("n", "f", "f*", "B", "B*", "b", "b*"):
            current = None
            segments = []
    if state_stack:
        raise VeritaSourceSignatureError("BU8 unterminated PDF graphics state")
    return paths, raw


def _validate_and_select_paths(
    config: Mapping[str, Any], paths: Mapping[int, StrokedPath]
) -> dict[str, dict[str, StrokedPath]]:
    selected: dict[str, dict[str, StrokedPath]] = {}
    for domain in DOMAINS:
        domain_config = config["vector_source"][domain]
        selected[domain] = {}
        expected_dash = domain_config["dash"]
        for channel in CHANNELS:
            spec = domain_config["channels"][channel]
            operation_index = int(spec["stroke_operation_index"])
            path = paths.get(operation_index)
            if path is None:
                raise VeritaSourceSignatureError(
                    f"BU8 missing {domain}/{channel} vector path"
                )
            cubic_count = sum(segment.kind == "cubic" for segment in path.segments)
            line_count = sum(segment.kind == "line" for segment in path.segments)
            if (
                cubic_count != int(spec["cubic_segments"])
                or line_count != int(spec.get("leading_lines", 0))
                or not math.isclose(
                    path.width, float(domain_config["line_width"]), abs_tol=0.001
                )
            ):
                raise VeritaSourceSignatureError(
                    f"BU8 {domain}/{channel} vector geometry drift"
                )
            if expected_dash == "solid":
                dash_ok = not path.dash
            else:
                dash_ok = path.dash == tuple(float(value) for value in expected_dash)
            if not dash_ok:
                raise VeritaSourceSignatureError(f"BU8 {domain}/{channel} dash drift")
            selected[domain][channel] = path
        probe_x = float(domain_config["label_probe_x_pdf"])
        observed_order = tuple(
            sorted(
                CHANNELS,
                key=lambda name: _path_value_at_x(selected[domain][name], probe_x),
                reverse=True,
            )
        )
        if observed_order != tuple(domain_config["channel_order_high_to_low_pdf_y"]):
            raise VeritaSourceSignatureError(f"BU8 {domain} source-label order drift")
    return selected


def _bezier(points: Sequence[tuple[float, float]], t: float) -> tuple[float, float]:
    one = 1.0 - t
    weights = (one**3, 3.0 * one * one * t, 3.0 * one * t * t, t**3)
    return (
        float(sum(weight * point[0] for weight, point in zip(weights, points))),
        float(sum(weight * point[1] for weight, point in zip(weights, points))),
    )


def _segment_value_at_x(segment: Segment, target_x: float) -> float:
    if segment.kind == "line":
        first, second = segment.points
        if math.isclose(first[0], second[0]):
            return float((first[1] + second[1]) * 0.5)
        fraction = (target_x - first[0]) / (second[0] - first[0])
        return float(first[1] + fraction * (second[1] - first[1]))
    low = 0.0
    high = 1.0
    increasing = segment.points[-1][0] >= segment.points[0][0]
    for _ in range(64):
        middle = (low + high) * 0.5
        x, _ = _bezier(segment.points, middle)
        if (x < target_x) == increasing:
            low = middle
        else:
            high = middle
    return _bezier(segment.points, (low + high) * 0.5)[1]


def _path_value_at_x(path: StrokedPath, target_x: float) -> float:
    for segment in path.segments:
        first_x = segment.points[0][0]
        last_x = segment.points[-1][0]
        if min(first_x, last_x) - 1e-8 <= target_x <= max(first_x, last_x) + 1e-8:
            return _segment_value_at_x(segment, target_x)
    low = min(segment.points[0][0] for segment in path.segments)
    high = max(segment.points[-1][0] for segment in path.segments)
    raise VeritaSourceSignatureError(
        f"BU8 vector query {target_x:.6f} outside path [{low:.6f}, {high:.6f}]"
    )


def _axis_coordinate(value: float, axis: Mapping[str, Any]) -> float:
    (value0, coordinate0), (value1, coordinate1) = axis["anchors"]
    if axis["scale"] == "log10":
        fraction = (math.log10(value) - math.log10(value0)) / (
            math.log10(value1) - math.log10(value0)
        )
    else:
        fraction = (value - value0) / (value1 - value0)
    return float(coordinate0 + fraction * (coordinate1 - coordinate0))


def _axis_value(coordinate: float, axis: Mapping[str, Any]) -> float:
    (value0, coordinate0), (value1, coordinate1) = axis["anchors"]
    fraction = (coordinate - coordinate0) / (coordinate1 - coordinate0)
    if axis["scale"] == "log10":
        return float(
            10.0
            ** (
                math.log10(value0)
                + fraction * (math.log10(value1) - math.log10(value0))
            )
        )
    return float(value0 + fraction * (value1 - value0))


def _sample_verita_domain(
    domain_config: Mapping[str, Any],
    paths: Mapping[str, StrokedPath],
    coordinates: Sequence[float],
    *,
    source_exposure_offset: float = 0.0,
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for channel in CHANNELS:
        values: list[float] = []
        for coordinate in coordinates:
            source_coordinate = float(coordinate) - source_exposure_offset
            page_x = _axis_coordinate(source_coordinate, domain_config["x_axis"])
            page_y = _path_value_at_x(paths[channel], page_x)
            values.append(_axis_value(page_y, domain_config["y_axis"]))
        result[channel] = {
            "coordinates": [float(value) for value in coordinates],
            "values": values,
        }
    return result


def _source_value_from_pixel(
    pixel: float, anchors: Sequence[Sequence[float]], *, logarithmic: bool
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    if logarithmic:
        return float(
            10.0
            ** (
                math.log10(float(value0))
                + fraction * (math.log10(float(value1)) - math.log10(float(value0)))
            )
        )
    return float(value0 + fraction * (value1 - value0))


def _load_vision3_curves(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    evidence = config["vision3_evidence"]
    for name, row in evidence.items():
        path = root / _relative_path(str(row["path"]))
        if not path.is_file() or hash_file(path) != row["sha256"]:
            raise VeritaSourceSignatureError(f"BU8 parent evidence drift: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            "required_decision" in row
            and payload.get("decision") != row["required_decision"]
        ):
            raise VeritaSourceSignatureError(f"BU8 parent decision drift: {name}")
        loaded[name] = payload
        hashes[name] = row["sha256"]
    return loaded, hashes


def _vision3_domain_samples(
    trace: Mapping[str, Any],
    coordinates: Sequence[float],
    *,
    domain: str,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    stocks: dict[str, dict[str, dict[str, list[float]]]] = {}
    for stock in VISION3_STOCKS:
        row = trace["stocks"][stock]
        channels: dict[str, dict[str, list[float]]] = {}
        axes = row["graph_axes"]
        for channel in CHANNELS:
            points = np.asarray(row["curves"][channel], dtype=np.float64)
            x_values = np.asarray(
                [
                    _source_value_from_pixel(
                        value,
                        axes["x_value_pixels"],
                        logarithmic=axes["x_scale"] == "log10",
                    )
                    for value in points[:, 0]
                ],
                dtype=np.float64,
            )
            y_values = np.asarray(
                [
                    _source_value_from_pixel(
                        value,
                        axes["y_value_pixels"],
                        logarithmic=axes["y_scale"] == "log10",
                    )
                    for value in points[:, 1]
                ],
                dtype=np.float64,
            )
            if domain == "characteristic":
                channels[channel] = {
                    "coordinates": x_values.tolist(),
                    "values": y_values.tolist(),
                }
                continue
            query = np.asarray(coordinates, dtype=np.float64)
            if domain == "mtf":
                sampled = np.exp(
                    np.interp(np.log(query), np.log(x_values), np.log(y_values))
                )
            elif domain == "granularity":
                sampled = np.exp(np.interp(query, x_values, np.log(y_values)))
            else:
                sampled = np.interp(query, x_values, y_values)
            channels[channel] = {
                "coordinates": query.tolist(),
                "values": sampled.tolist(),
            }
        stocks[stock] = channels
    return stocks


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(y >= level)
    if not indices.size:
        raise VeritaSourceSignatureError("BU8 characteristic crossing is absent")
    index = int(indices[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if math.isclose(y0, y1):
        return x1
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _characteristic_shape(
    coordinates: Sequence[float], values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = np.asarray(coordinates, dtype=np.float64)
    raw = np.asarray(values, dtype=np.float64)
    monotone = np.maximum.accumulate(raw)
    span = float(monotone[-1] - monotone[0])
    if span <= 0.0:
        raise VeritaSourceSignatureError("BU8 characteristic density span is empty")
    normalized = (monotone - monotone[0]) / span
    x10 = _crossing(x, normalized, 0.1)
    x90 = _crossing(x, normalized, 0.9)
    if x90 <= x10:
        raise VeritaSourceSignatureError("BU8 characteristic gauge is invalid")
    x_normalized = (x - x10) / (x90 - x10)
    reversal = float(np.max(np.maximum(0.0, raw[:-1] - raw[1:])))
    return (
        x_normalized,
        normalized,
        {
            "density_span": span,
            "maximum_monotonic_reversal_density": reversal,
            "x10": x10,
            "x90": x90,
            "x10_to_x90_span": x90 - x10,
        },
    )


def _compare_characteristic(
    config: Mapping[str, Any],
    verita: Mapping[str, Any],
    vision3: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = config["comparison"]["characteristic"]
    low_bound, high_bound, samples = comparison["comparison_grid"]
    gate = float(comparison["material_threshold"])
    verita_shapes = {
        channel: _characteristic_shape(
            verita[channel]["coordinates"], verita[channel]["values"]
        )
        for channel in CHANNELS
    }
    pairwise: dict[str, Any] = {}
    material_pairs: list[str] = []
    for stock in VISION3_STOCKS:
        channel_rows: dict[str, Any] = {}
        material_count = 0
        for channel in CHANNELS:
            source_shape = _characteristic_shape(
                vision3[stock][channel]["coordinates"],
                vision3[stock][channel]["values"],
            )
            first_x, first_y, _ = verita_shapes[channel]
            second_x, second_y, _ = source_shape
            low = max(float(first_x[0]), float(second_x[0]), float(low_bound))
            high = min(float(first_x[-1]), float(second_x[-1]), float(high_bound))
            if high <= low:
                raise VeritaSourceSignatureError(
                    "BU8 empty characteristic comparison domain"
                )
            grid = np.linspace(low, high, int(samples), dtype=np.float64)
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.interp(grid, first_x, first_y)
                            - np.interp(grid, second_x, second_y)
                        )
                    )
                )
            )
            material = rmse >= gate
            material_count += int(material)
            channel_rows[channel] = {
                "normalized_shape_rmse": rmse,
                "material": material,
                "comparison_domain": [low, high],
            }
        pair_id = f"{STOCK}__vs__{stock}"
        pair_material = material_count >= int(
            config["gates"]["minimum_material_channels_per_domain_pair"]
        )
        if pair_material:
            material_pairs.append(pair_id)
        pairwise[pair_id] = {
            "channels": channel_rows,
            "material_channel_count": material_count,
            "material_domain_pair": pair_material,
        }
    diagnostics = {channel: verita_shapes[channel][2] for channel in CHANNELS}
    return pairwise, {"material_pairs": material_pairs, "verita": diagnostics}


def _compare_log_domain(
    config: Mapping[str, Any],
    verita: Mapping[str, Any],
    vision3: Mapping[str, Any],
    *,
    domain: str,
) -> tuple[dict[str, Any], list[str]]:
    gate = float(config["comparison"][domain]["material_threshold"])
    pairwise: dict[str, Any] = {}
    material_pairs: list[str] = []
    for stock in VISION3_STOCKS:
        channel_rows: dict[str, Any] = {}
        material_count = 0
        for channel in CHANNELS:
            first = np.log(np.asarray(verita[channel]["values"], dtype=np.float64))
            second = np.log(
                np.asarray(vision3[stock][channel]["values"], dtype=np.float64)
            )
            if domain == "mtf":
                first = first - first[0]
                second = second - second[0]
            else:
                first = first - float(np.mean(first))
                second = second - float(np.mean(second))
            rmse = float(np.sqrt(np.mean(np.square(first - second))))
            material = rmse >= gate
            material_count += int(material)
            channel_rows[channel] = {
                "normalized_log_shape_rmse": rmse,
                "material": material,
            }
        pair_id = f"{STOCK}__vs__{stock}"
        pair_material = material_count >= int(
            config["gates"]["minimum_material_channels_per_domain_pair"]
        )
        if pair_material:
            material_pairs.append(pair_id)
        pairwise[pair_id] = {
            "channels": channel_rows,
            "material_channel_count": material_count,
            "material_domain_pair": pair_material,
        }
    return pairwise, material_pairs


def _draw_overlay(
    output: Path,
    samples: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
) -> str:
    width, height = 1500, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colours = {"blue": (25, 80, 220), "green": (25, 145, 65), "red": (210, 45, 45)}
    for domain_index, domain in enumerate(DOMAINS):
        left = 35 + domain_index * 500
        top = 35
        plot_width = 430
        plot_height = 420
        draw.rectangle(
            (left, top, left + plot_width, top + plot_height),
            outline=(45, 45, 45),
            width=2,
        )
        draw.text((left + 8, top + 8), domain, fill=(10, 10, 10))
        values = samples[domain]
        all_x = [value for row in values.values() for value in row["coordinates"]]
        all_y = [value for row in values.values() for value in row["values"]]
        x_min, x_max = min(all_x), max(all_x)
        if domain in ("mtf", "granularity"):
            plotted_y = [math.log10(value) for value in all_y]
        else:
            plotted_y = all_y
        y_min, y_max = min(plotted_y), max(plotted_y)
        for channel in CHANNELS:
            row = values[channel]
            points: list[tuple[float, float]] = []
            for x_value, y_value in zip(row["coordinates"], row["values"]):
                x_fraction = (float(x_value) - x_min) / (x_max - x_min)
                y_plot = (
                    math.log10(float(y_value))
                    if domain in ("mtf", "granularity")
                    else float(y_value)
                )
                y_fraction = (y_plot - y_min) / max(y_max - y_min, 1e-12)
                points.append(
                    (
                        left + x_fraction * plot_width,
                        top + (1.0 - y_fraction) * plot_height,
                    )
                )
            draw.line(points, fill=colours[channel], width=3)
            for point in points:
                draw.ellipse(
                    (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                    fill=colours[channel],
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return hash_file(output)


def audit_source_signature(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> dict[str, Any]:
    source = config["source"]
    source_path = root / _relative_path(str(source["path"]))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(source["expected_bytes"])
        or hash_file(source_path) != source["sha256"]
    ):
        raise VeritaSourceSignatureError("BU8 VERITA source integrity mismatch")
    reader = PdfReader(str(source_path))
    if len(reader.pages) != int(config["gates"]["required_source_pages"]):
        raise VeritaSourceSignatureError("BU8 VERITA page count drift")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = " ".join(text.split())
    text_gate = all(
        " ".join(str(anchor).split()) in normalized_text
        for anchor in config["required_text_anchors"]
    )
    page = reader.pages[int(config["vector_source"]["page_zero_based"])]
    paths, content = _extract_stroked_paths(page)
    if (
        len(content) != int(config["vector_source"]["decoded_content_bytes"])
        or hashlib.sha256(content).hexdigest()
        != config["vector_source"]["decoded_content_sha256"]
    ):
        raise VeritaSourceSignatureError("BU8 decoded PDF content drift")
    selected = _validate_and_select_paths(config, paths)
    parents, parent_hashes = _load_vision3_curves(root, config)

    characteristic_coordinates = config["comparison"]["characteristic"][
        "sample_log_relative_exposures"
    ]
    mtf_coordinates = config["comparison"]["mtf"]["common_frequencies_cycles_per_mm"]
    granularity_coordinates = config["comparison"]["granularity"][
        "common_log_relative_exposures"
    ]
    verita = {
        "characteristic": _sample_verita_domain(
            config["vector_source"]["characteristic"],
            selected["characteristic"],
            characteristic_coordinates,
            source_exposure_offset=3.0,
        ),
        "mtf": _sample_verita_domain(
            config["vector_source"]["mtf"], selected["mtf"], mtf_coordinates
        ),
        "granularity": _sample_verita_domain(
            config["vector_source"]["granularity"],
            selected["granularity"],
            granularity_coordinates,
            source_exposure_offset=3.0,
        ),
    }
    vision3 = {
        "characteristic": _vision3_domain_samples(
            parents["characteristic_trace"],
            characteristic_coordinates,
            domain="characteristic",
        ),
        "mtf": _vision3_domain_samples(
            parents["mtf_trace"], mtf_coordinates, domain="mtf"
        ),
        "granularity": _vision3_domain_samples(
            parents["granularity_trace"],
            granularity_coordinates,
            domain="granularity",
        ),
    }
    characteristic_pairwise, characteristic_summary = _compare_characteristic(
        config, verita["characteristic"], vision3["characteristic"]
    )
    mtf_pairwise, mtf_material = _compare_log_domain(
        config, verita["mtf"], vision3["mtf"], domain="mtf"
    )
    granularity_pairwise, granularity_material = _compare_log_domain(
        config,
        verita["granularity"],
        vision3["granularity"],
        domain="granularity",
    )
    domain_pairwise = {
        "characteristic": characteristic_pairwise,
        "mtf": mtf_pairwise,
        "granularity": granularity_pairwise,
    }
    material_domains_by_pair: dict[str, list[str]] = {}
    material_stock_pairs: list[str] = []
    for stock in VISION3_STOCKS:
        pair_id = f"{STOCK}__vs__{stock}"
        domains = [
            domain
            for domain in DOMAINS
            if domain_pairwise[domain][pair_id]["material_domain_pair"]
        ]
        material_domains_by_pair[pair_id] = domains
        if len(domains) >= int(
            config["gates"]["minimum_material_domains_per_vision3_pair"]
        ):
            material_stock_pairs.append(pair_id)

    characteristic_gate = all(
        diagnostics["maximum_monotonic_reversal_density"]
        <= float(config["gates"]["maximum_characteristic_monotonic_reversal_density"])
        and diagnostics["density_span"]
        >= float(config["gates"]["minimum_characteristic_density_span"])
        and diagnostics["x10_to_x90_span"]
        >= float(config["gates"]["minimum_characteristic_10_to_90_exposure_span"])
        for diagnostics in characteristic_summary["verita"].values()
    )
    mtf_values = np.asarray(
        [value for channel in CHANNELS for value in verita["mtf"][channel]["values"]],
        dtype=np.float64,
    )
    mtf_increases = [
        float(
            np.max(
                np.maximum(
                    0.0,
                    np.diff(np.asarray(verita["mtf"][channel]["values"]))
                    / np.asarray(verita["mtf"][channel]["values"][:-1])
                    * 100.0,
                )
            )
        )
        for channel in CHANNELS
    ]
    mtf_gate = bool(
        np.all(np.isfinite(mtf_values))
        and float(np.min(mtf_values))
        >= float(config["gates"]["minimum_mtf_response_percent"])
        and float(np.max(mtf_values))
        <= float(config["gates"]["maximum_mtf_response_percent"])
        and max(mtf_increases)
        <= float(config["gates"]["maximum_mtf_monotonic_increase_percent"])
    )
    granularity_values = np.asarray(
        [
            value
            for channel in CHANNELS
            for value in verita["granularity"][channel]["values"]
        ],
        dtype=np.float64,
    )
    granularity_gate = bool(
        np.all(np.isfinite(granularity_values))
        and float(np.min(granularity_values))
        >= float(config["gates"]["minimum_granularity_sigma_d"])
        and float(np.max(granularity_values))
        <= float(config["gates"]["maximum_granularity_sigma_d"])
    )
    overlay_sha256 = _draw_overlay(overlay_path, verita)
    signature_gate = len(material_stock_pairs) >= int(
        config["gates"]["minimum_material_vision3_pairs"]
    )
    gate_results = {
        "source_integrity": True,
        "first_party_text_anchors": text_gate,
        "exact_vector_content": True,
        "three_domains_three_channels": len(selected)
        == int(config["gates"]["required_vector_domains"])
        and all(
            len(selected[domain])
            == int(config["gates"]["required_channels_per_domain"])
            for domain in DOMAINS
        ),
        "vision3_parent_evidence_exact": len(parent_hashes)
        == len(config["vision3_evidence"]),
        "characteristic_source_range": characteristic_gate,
        "mtf_source_range_and_monotonicity": mtf_gate,
        "granularity_source_range": granularity_gate,
        "multi_domain_stock_signature": signature_gate,
        "visual_overlay_generated": bool(overlay_sha256),
        "no_operator_fit_or_photographic_pixels": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source": {
            "pdf_sha256": source["sha256"],
            "pdf_bytes": int(source["expected_bytes"]),
            "pdf_pages": len(reader.pages),
            "decoded_content_sha256": hashlib.sha256(content).hexdigest(),
            "stock_id": STOCK,
            "measurement_context": source["measurement_context"],
        },
        "parent_evidence_sha256": parent_hashes,
        "verita_samples": verita,
        "characteristic_diagnostics": characteristic_summary["verita"],
        "mtf_maximum_monotonic_increase_percent": dict(zip(CHANNELS, mtf_increases)),
        "pairwise": domain_pairwise,
        "material_pairs_by_domain": {
            "characteristic": characteristic_summary["material_pairs"],
            "mtf": mtf_material,
            "granularity": granularity_material,
        },
        "material_domains_by_pair": material_domains_by_pair,
        "material_stock_pairs": material_stock_pairs,
        "overlay_sha256": overlay_sha256,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "signature_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "decision": (
            config["branch_rule"]["pass"] if passed else config["branch_rule"]["fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "CHANNELS",
    "DOMAINS",
    "REPORT_SCHEMA",
    "SCHEMA",
    "STOCK",
    "VISION3_STOCKS",
    "VeritaSourceSignatureError",
    "audit_source_signature",
    "canonical_json",
    "load_contract",
]
