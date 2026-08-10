"""U5.R2CB1 Fujifilm E-6 spectral dye source-signature audit."""

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

SCHEMA = "neuro_film.u5_r2cb1_fujifilm_e6_spectral_dye_signature_contract.v1"
TRACE_SCHEMA = "neuro_film.fujifilm_e6_spectral_dye_density_trace.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb1_fujifilm_e6_spectral_dye_signature_report.v1"
EXPERIMENT_ID = "U5.R2CB1"
STOCKS = (
    "fujichrome_velvia_50_rvp50_af3_0221e2",
    "fujichrome_velvia_100_rvp100_af3_202e",
    "fujichrome_provia_100f_rdpiii_af3_036e",
)
CHANNELS = ("yellow", "magenta", "cyan")
CONTRACT_SHA256 = "4aa78a31d73e2452b1a9a8e111ea30af40284fa68d3284c641fcf2764bf21d62"


class FujifilmSpectralDyeError(RuntimeError):
    """Raised when a frozen CB1 input or contract drifts."""


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
        raise FujifilmSpectralDyeError("CB1 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmSpectralDyeError("CB1 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or tuple(payload.get("stocks", ())) != STOCKS
        or tuple(comparison.get("channels", ())) != CHANNELS
        or comparison.get("common_wavelengths_nm", {}).get("yellow")
        != [410.0, 430.0, 450.0, 470.0, 490.0, 510.0, 530.0, 550.0, 570.0, 590.0]
    ):
        raise FujifilmSpectralDyeError("CB1 frozen contract drift")
    for key in ("trace", "parent_source_signature"):
        _relative_path(str(payload[key]["path"]))
    return payload


def _linear_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if pixel0 == pixel1:
        raise FujifilmSpectralDyeError("CB1 invalid linear axis anchors")
    fraction = (float(pixel) - float(pixel0)) / (float(pixel1) - float(pixel0))
    return float(float(value0) + fraction * (float(value1) - float(value0)))


def _axis_residual(
    ticks: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    return float(
        max(
            abs(
                float(pixel)
                - (
                    float(pixel0)
                    + (float(value) - float(value0))
                    / (float(value1) - float(value0))
                    * (float(pixel1) - float(pixel0))
                )
            )
            for value, pixel in ticks
        )
    )


def _curve_values(row: Mapping[str, Any], channel: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(row["curves"][channel], dtype=np.float64)
    axes = row["graph_axes"]
    wavelengths = np.asarray(
        [_linear_value_from_pixel(x, axes["x_value_pixels"]) for x in points[:, 0]]
    )
    densities = np.asarray(
        [_linear_value_from_pixel(y, axes["y_value_pixels"]) for y in points[:, 1]]
    )
    if not np.all(np.diff(wavelengths) > 0.0):
        raise FujifilmSpectralDyeError("CB1 trace wavelengths must increase")
    return wavelengths, densities


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] < 128:
                best = min(best, math.hypot(xx - x, yy - y))
    return float(best if math.isfinite(best) else maximum + 1.0)


def _draw_overlay(row: Mapping[str, Any], graph: Path, output: Path) -> str:
    image = Image.open(graph).convert("RGB")
    draw = ImageDraw.Draw(image)
    colours = {"yellow": (255, 190, 0), "magenta": (255, 30, 180), "cyan": (0, 210, 255)}
    for channel in CHANNELS:
        xy = [tuple(map(int, point)) for point in row["curves"][channel]]
        draw.line(xy, fill=colours[channel], width=2)
        for x, y in xy:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=colours[channel], width=1)
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
        return _verify_embedded_graph(
            root / _relative_path(row["source_pdf"]["path"]),
            graph,
            int(graph_info["page_one_based"]),
            int(graph_info["embedded_page_image_index_zero_based"]),
        )
    derivation = graph_info["derivation"]
    full = root / _relative_path(derivation["full_page_render_path"])
    if not full.is_file() or hash_file(full) != derivation["full_page_render_sha256"]:
        return False
    full_image = Image.open(full)
    if full_image.size != (
        int(derivation["full_page_render_width"]),
        int(derivation["full_page_render_height"]),
    ):
        return False
    crop = full_image.crop(tuple(map(int, derivation["crop_box_left_top_right_bottom"])))
    return bool(np.array_equal(np.asarray(crop), np.asarray(Image.open(graph))))


def _stable_identity(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("stable_evidence_id", None)
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def audit_signature(
    config: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    gates = config["gates"]
    trace_path = root / _relative_path(config["trace"]["path"])
    parent_path = root / _relative_path(config["parent_source_signature"]["path"])
    for path, expected in (
        (trace_path, config["trace"]["sha256"]),
        (parent_path, config["parent_source_signature"]["sha256"]),
    ):
        if not path.is_file() or hash_file(path) != expected:
            raise FujifilmSpectralDyeError(f"CB1 input integrity mismatch: {path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if trace.get("schema") != TRACE_SCHEMA or tuple(trace.get("stocks", {})) != STOCKS:
        raise FujifilmSpectralDyeError("CB1 trace structure drift")
    if parent.get("decision") != config["parent_source_signature"]["required_decision"]:
        raise FujifilmSpectralDyeError("CB1 parent source signature missing")

    common = {
        channel: np.asarray(config["comparison"]["common_wavelengths_nm"][channel])
        for channel in CHANNELS
    }
    samples: dict[str, dict[str, np.ndarray]] = {}
    stocks: dict[str, Any] = {}
    density_slopes: dict[str, float] = {}
    source_gate = derivation_gate = axis_gate = ink_gate = density_gate = context_gate = True
    point_count_gate = True
    for stock in STOCKS:
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
            raise FujifilmSpectralDyeError(f"CB1 source integrity mismatch: {stock}")
        source_gate &= source_ok
        derivation_ok = _verify_graph_derivation(root, row, graph)
        derivation_gate &= derivation_ok
        axes = row["graph_axes"]
        x_residual = _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"])
        y_residual = _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"])
        axis_gate &= max(x_residual, y_residual) <= float(gates["axis_max_residual_px"])
        image = np.asarray(Image.open(graph).convert("L"))
        samples[stock] = {}
        channel_rows: dict[str, Any] = {}
        for channel in CHANNELS:
            points = row["curves"][channel]
            point_count_gate &= len(points) >= int(gates["minimum_trace_points_per_channel"])
            ink_distances = [
                _ink_distance(image, int(x), int(y), float(gates["trace_max_ink_distance_px"]))
                for x, y in points
            ]
            ink_gate &= max(ink_distances) <= float(gates["trace_max_ink_distance_px"])
            wavelengths, densities = _curve_values(row, channel)
            density_gate &= bool(
                densities.min() >= float(gates["minimum_density"])
                and densities.max() <= float(gates["maximum_density"])
            )
            if (
                common[channel][0] < wavelengths[0] - 1.0
                or common[channel][-1] > wavelengths[-1] + 1.0
            ):
                raise FujifilmSpectralDyeError("CB1 common wavelengths exceed trace support")
            samples[stock][channel] = np.interp(common[channel], wavelengths, densities)
            channel_rows[channel] = {
                "densities": densities.tolist(),
                "ink_max_distance_px": max(ink_distances),
                "sampled_densities": samples[stock][channel].tolist(),
                "wavelengths_nm": wavelengths.tolist(),
            }
        context = row["measurement_context"]
        context_gate &= bool(
            context["process"] == gates["required_process"]
            and context["exposure"] == gates["required_exposure"]
            and context["density_unit"] == gates["required_density_unit"]
        )
        y_anchors = axes["y_value_pixels"]
        density_slopes[stock] = abs(
            (float(y_anchors[1][0]) - float(y_anchors[0][0]))
            / (float(y_anchors[1][1]) - float(y_anchors[0][1]))
        )
        stocks[stock] = {
            "axis_max_residual_px": max(x_residual, y_residual),
            "channels": channel_rows,
            "derivation_verified": derivation_ok,
            "overlay_sha256": _draw_overlay(
                row, graph, overlay_dir / f"{stock}_spectral_dye_overlay.png"
            ),
            "source_verified": source_ok,
        }

    pairs: dict[str, Any] = {}
    material_count = 0
    uncertainty_gate = True
    pair_gate = True
    for index, first in enumerate(STOCKS):
        for second in STOCKS[index + 1 :]:
            first_values = np.concatenate([samples[first][channel] for channel in CHANNELS])
            second_values = np.concatenate([samples[second][channel] for channel in CHANNELS])
            rmse = float(np.sqrt(np.mean((first_values - second_values) ** 2)))
            uncertainty = float(gates["digitization_uncertainty_px"]) * (
                density_slopes[first] + density_slopes[second]
            )
            fraction = uncertainty / rmse if rmse > 0.0 else math.inf
            material = rmse >= float(gates["minimum_each_pair_density_rmse"])
            pair_uncertainty_ok = fraction <= float(
                gates["maximum_uncertainty_fraction_of_observed_pair_separation"]
            )
            material_count += int(material)
            pair_gate &= material
            uncertainty_gate &= pair_uncertainty_ok
            pairs[f"{first}__{second}"] = {
                "density_rmse": rmse,
                "digitization_uncertainty_bound": uncertainty,
                "digitization_uncertainty_fraction": fraction,
                "material": material,
                "uncertainty_gate": pair_uncertainty_ok,
            }

    gate_results = {
        "axis_calibration": axis_gate,
        "density_range": density_gate,
        "graph_derivation": derivation_gate,
        "like_for_like_measurement_context": context_gate,
        "material_all_stock_pairs": pair_gate
        and material_count == int(gates["required_material_pair_count"]),
        "point_count": point_count_gate,
        "source_integrity": source_gate,
        "trace_ink": ink_gate,
        "uncertainty_fraction": uncertainty_gate,
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "claim_ceiling": config["claim_ceiling"],
        "decision": "retain_non_renderable_fujifilm_e6_spectral_dye_signature"
        if passed
        else "close_exact_fujifilm_e6_spectral_dye_signature",
        "failed_gates": [key for key, value in gate_results.items() if not value],
        "gate_results": gate_results,
        "material_pair_count": material_count,
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
