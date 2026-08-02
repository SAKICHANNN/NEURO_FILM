"""U5.R2BW0 Fujifilm Velvia 50 MTF revision/source-signature audit."""

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

SCHEMA = "neuro_film.u5_r2bw0_fujifilm_velvia_mtf_revision_signature_contract.v1"
TRACE_SCHEMA = "neuro_film.fujifilm_velvia50_mtf_revision_trace.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bw0_fujifilm_velvia_mtf_revision_signature_report.v1"
EXPERIMENT_ID = "U5.R2BW0"
REVISIONS = (
    "fujichrome_velvia_rvp_af3_960e",
    "fujichrome_velvia_50_rvp50_af3_0221e2",
)
KODAK_STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
CHANNELS = ("blue", "green", "red")


class VelviaMtfSignatureError(RuntimeError):
    """Raised when a frozen BW0 input or contract drifts."""


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
        raise VelviaMtfSignatureError("BW0 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("trace", {}).get("sha256")
        != "d5e1d7bddbfa434434cb092fce438355673bb08173c5cc470600c39fbc1fc481"
        or payload.get("kodak_trace", {}).get("sha256")
        != "d88c0e698c0e0969696224df50f1573ed78a6de0a894075f6502d9b3f658a794"
        or payload.get("parent_observed_profile", {}).get("sha256")
        != "d4c318a857f788b3ea5fdef292ba44f24f2d4386a03cb02928c77a24424c95a9"
        or tuple(comparison.get("velvia_revisions", ())) != REVISIONS
        or tuple(comparison.get("kodak_stocks", ())) != KODAK_STOCKS
        or comparison.get("current_revision") != REVISIONS[1]
        or comparison.get("common_frequencies_cycles_per_mm")
        != [25.0, 35.0, 45.0, 60.0]
        or comparison.get("kodak_composite")
        != "geometric mean of blue, green and red response at each common frequency"
        or comparison.get("normalization")
        != "subtract natural-log response at 25 cycles/mm"
        or comparison.get("interpolation")
        != "linear in log10 spatial frequency and natural-log response"
        or gates
        != {
            "required_revision_count": 2,
            "minimum_trace_points_per_revision": 10,
            "axis_max_residual_px": 2.0,
            "trace_max_ink_distance_px": 2.0,
            "minimum_response_percent": 20.0,
            "maximum_response_percent": 140.0,
            "maximum_revision_log_shape_rmse": 0.03,
            "minimum_each_kodak_log_shape_rmse": 0.08,
            "digitization_uncertainty_px": 2.0,
            "maximum_uncertainty_fraction_of_separation_gate": 0.5,
            "require_equal_granularity_observation": True,
            "required_granularity_aperture_micrometres": 48,
            "required_granularity_sample_density_above_minimum": 1.0,
            "two_byte_identical_audits": True,
            "visual_overlay_required": True,
        }
    ):
        raise VelviaMtfSignatureError("BW0 frozen contract drift")
    for key in ("trace", "kodak_trace", "parent_observed_profile"):
        _relative_path(str(payload[key]["path"]))
    return payload


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != TRACE_SCHEMA
        or tuple(payload.get("revisions", {})) != REVISIONS
    ):
        raise VelviaMtfSignatureError("BW0 trace contract drift")
    for revision in REVISIONS:
        row = payload["revisions"][revision]
        for key in ("source_pdf", "source_graph"):
            _relative_path(str(row[key]["path"]))
        if (
            row.get("graph_axes", {}).get("x_scale") != "log10"
            or row.get("graph_axes", {}).get("y_scale") != "log10"
        ):
            raise VelviaMtfSignatureError(f"BW0 axis type drift: {revision}")
    return payload


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 <= 0.0 or value1 <= 0.0 or pixel0 == pixel1:
        raise VelviaMtfSignatureError("BW0 invalid logarithmic axis anchors")
    fraction = (float(pixel) - float(pixel0)) / (float(pixel1) - float(pixel0))
    return float(
        10.0
        ** (
            math.log10(float(value0))
            + fraction * (math.log10(float(value1)) - math.log10(float(value0)))
        )
    )


def _axis_residual(
    rows: Sequence[Sequence[float]], anchors: Sequence[Sequence[float]]
) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    denominator = math.log10(float(value1)) - math.log10(float(value0))
    return float(
        max(
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
        )
    )


def _ink_distance(image: np.ndarray, x: int, y: int, maximum: float) -> float:
    radius = math.ceil(maximum)
    best = math.inf
    for yy in range(max(0, y - radius), min(image.shape[0], y + radius + 1)):
        for xx in range(max(0, x - radius), min(image.shape[1], x + radius + 1)):
            if image[yy, xx] > 127:
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


def _curve_values(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axes = row["graph_axes"]
    points = np.asarray(row["curve"], dtype=np.float64)
    frequencies = np.asarray(
        [_log_value_from_pixel(x, axes["x_value_pixels"]) for x in points[:, 0]],
        dtype=np.float64,
    )
    responses = np.asarray(
        [_log_value_from_pixel(y, axes["y_value_pixels"]) for y in points[:, 1]],
        dtype=np.float64,
    )
    if not np.all(np.diff(frequencies) > 0.0):
        raise VelviaMtfSignatureError("BW0 trace frequencies must be increasing")
    return frequencies, responses


def _normalized_log_curve(
    frequencies: np.ndarray, responses: np.ndarray, common: np.ndarray
) -> np.ndarray:
    if common[0] < frequencies[0] or common[-1] > frequencies[-1]:
        raise VelviaMtfSignatureError("BW0 common frequencies exceed trace support")
    values = np.interp(np.log10(common), np.log10(frequencies), np.log(responses))
    return values - values[0]


def _kodak_composites(
    payload: Mapping[str, Any], common: np.ndarray
) -> dict[str, np.ndarray]:
    stocks = payload.get("stocks", {})
    if tuple(stocks) != KODAK_STOCKS:
        raise VelviaMtfSignatureError("BW0 Kodak stock order drift")
    result: dict[str, np.ndarray] = {}
    for stock in KODAK_STOCKS:
        row = stocks[stock]
        channel_values: list[np.ndarray] = []
        for channel in CHANNELS:
            points = np.asarray(row["curves"][channel], dtype=np.float64)
            axes = row["graph_axes"]
            frequencies = np.asarray(
                [_log_value_from_pixel(x, axes["x_value_pixels"]) for x in points[:, 0]]
            )
            responses = np.asarray(
                [_log_value_from_pixel(y, axes["y_value_pixels"]) for y in points[:, 1]]
            )
            channel_values.append(
                np.interp(np.log10(common), np.log10(frequencies), np.log(responses))
            )
        composite_log = np.mean(np.stack(channel_values, axis=0), axis=0)
        result[stock] = composite_log - composite_log[0]
    return result


def _stable_identity(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("report_sha256", None)
    stable.pop("stable_evidence_id", None)
    stable["revisions"] = {
        key: {k: v for k, v in value.items() if k != "overlay_sha256"}
        for key, value in stable["revisions"].items()
    }
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def audit_signature(
    config: Mapping[str, Any], root: Path, *, overlay_dir: Path
) -> dict[str, Any]:
    gates = config["gates"]
    trace_path = root / _relative_path(config["trace"]["path"])
    kodak_path = root / _relative_path(config["kodak_trace"]["path"])
    parent_path = root / _relative_path(config["parent_observed_profile"]["path"])
    for path, expected in (
        (trace_path, config["trace"]["sha256"]),
        (kodak_path, config["kodak_trace"]["sha256"]),
        (parent_path, config["parent_observed_profile"]["sha256"]),
    ):
        if not path.is_file() or hash_file(path) != expected:
            raise VelviaMtfSignatureError(f"BW0 parent integrity mismatch: {path}")

    trace = load_trace(trace_path)
    kodak = json.loads(kodak_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_gate = (
        parent.get("decision") == config["parent_observed_profile"]["required_decision"]
    )
    common = np.asarray(config["comparison"]["common_frequencies_cycles_per_mm"])

    revisions: dict[str, Any] = {}
    normalized: dict[str, np.ndarray] = {}
    source_gate = trace_gate = axis_gate = ink_gate = response_gate = True
    granularity_values: list[float] = []
    uncertainty_fractions: list[float] = []
    for revision in REVISIONS:
        row = trace["revisions"][revision]
        pdf = root / _relative_path(row["source_pdf"]["path"])
        graph = root / _relative_path(row["source_graph"]["path"])
        source_ok = bool(
            pdf.is_file()
            and pdf.stat().st_size == row["source_pdf"]["bytes"]
            and hash_file(pdf) == row["source_pdf"]["sha256"]
            and graph.is_file()
            and graph.stat().st_size == row["source_graph"]["bytes"]
            and hash_file(graph) == row["source_graph"]["sha256"]
        )
        if not source_ok:
            raise VelviaMtfSignatureError(f"BW0 source integrity mismatch: {revision}")
        source_gate &= source_ok
        embedded_ok = _verify_embedded_graph(
            pdf,
            graph,
            int(row["source_graph"]["page_one_based"]),
            int(row["source_graph"]["embedded_page_image_index_zero_based"]),
        )
        trace_gate &= embedded_ok
        image = np.asarray(Image.open(graph).convert("L"))
        if image.shape != (row["source_graph"]["height"], row["source_graph"]["width"]):
            raise VelviaMtfSignatureError(f"BW0 graph geometry mismatch: {revision}")
        axes = row["graph_axes"]
        x_residual = _axis_residual(axes["x_tick_value_pixels"], axes["x_value_pixels"])
        y_residual = _axis_residual(axes["y_tick_value_pixels"], axes["y_value_pixels"])
        axis_ok = max(x_residual, y_residual) <= gates["axis_max_residual_px"]
        axis_gate &= axis_ok
        ink_distances = [
            _ink_distance(image, int(x), int(y), gates["trace_max_ink_distance_px"])
            for x, y in row["curve"]
        ]
        ink_ok = max(ink_distances) <= gates["trace_max_ink_distance_px"]
        ink_gate &= ink_ok
        frequencies, responses = _curve_values(row)
        response_ok = bool(
            len(frequencies) >= gates["minimum_trace_points_per_revision"]
            and responses.min() >= gates["minimum_response_percent"]
            and responses.max() <= gates["maximum_response_percent"]
        )
        response_gate &= response_ok
        normalized[revision] = _normalized_log_curve(frequencies, responses, common)
        context = row["measurement_context"]
        granularity_values.append(float(context["diffuse_rms_granularity"]))
        y0, y1 = axes["y_value_pixels"]
        log_response_per_pixel = abs(
            (math.log(float(y1[0])) - math.log(float(y0[0])))
            / (float(y1[1]) - float(y0[1]))
        )
        normalized_uncertainty = (
            2.0 * gates["digitization_uncertainty_px"] * log_response_per_pixel
        )
        uncertainty_fractions.append(
            normalized_uncertainty / gates["minimum_each_kodak_log_shape_rmse"]
        )
        overlay = overlay_dir / f"{revision}_overlay.png"
        revisions[revision] = {
            "source_pdf_sha256": row["source_pdf"]["sha256"],
            "source_graph_sha256": row["source_graph"]["sha256"],
            "embedded_pixel_identity": embedded_ok,
            "axis_max_residual_px": max(x_residual, y_residual),
            "trace_max_ink_distance_px": max(ink_distances),
            "frequencies_cycles_per_mm": frequencies.tolist(),
            "responses_percent": responses.tolist(),
            "common_normalized_log_response": normalized[revision].tolist(),
            "diffuse_rms_granularity": context["diffuse_rms_granularity"],
            "granularity_aperture_micrometres": context[
                "granularity_aperture_micrometres"
            ],
            "overlay_sha256": _draw_overlay(graph, row["curve"], overlay),
        }

    revision_rmse = float(
        np.sqrt(np.mean(np.square(normalized[REVISIONS[0]] - normalized[REVISIONS[1]])))
    )
    kodak_curves = _kodak_composites(kodak, common)
    separations = {
        stock: float(np.sqrt(np.mean(np.square(normalized[REVISIONS[1]] - curve))))
        for stock, curve in kodak_curves.items()
    }
    granularity_gate = bool(
        gates["require_equal_granularity_observation"]
        and len(set(granularity_values)) == 1
        and all(
            trace["revisions"][revision]["measurement_context"][
                "granularity_aperture_micrometres"
            ]
            == gates["required_granularity_aperture_micrometres"]
            and trace["revisions"][revision]["measurement_context"][
                "granularity_sample_density_above_minimum"
            ]
            == gates["required_granularity_sample_density_above_minimum"]
            for revision in REVISIONS
        )
    )
    gate_results = {
        "parent_observed_profile": parent_gate,
        "source_integrity": source_gate,
        "embedded_graph_pixel_identity": trace_gate,
        "axis_calibration": axis_gate,
        "source_ink_trace": ink_gate,
        "trace_count_and_response_range": response_gate,
        "revision_stability": revision_rmse <= gates["maximum_revision_log_shape_rmse"],
        "each_kodak_stock_separation": all(
            value >= gates["minimum_each_kodak_log_shape_rmse"]
            for value in separations.values()
        ),
        "digitization_uncertainty": max(uncertainty_fractions)
        <= gates["maximum_uncertainty_fraction_of_separation_gate"],
        "granularity_observation_stability": granularity_gate,
        "visual_overlays_emitted": True,
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        "trace_sha256": config["trace"]["sha256"],
        "kodak_trace_sha256": config["kodak_trace"]["sha256"],
        "parent_observed_profile_sha256": config["parent_observed_profile"]["sha256"],
        "common_frequencies_cycles_per_mm": common.tolist(),
        "revisions": revisions,
        "revision_log_shape_rmse": revision_rmse,
        "kodak_composite_normalized_log_response": {
            key: value.tolist() for key, value in kodak_curves.items()
        },
        "current_velvia_to_kodak_log_shape_rmse": separations,
        "maximum_digitization_uncertainty_fraction_of_separation_gate": max(
            uncertainty_fractions
        ),
        "gate_results": gate_results,
        "passed": passed,
        "decision": (
            "retain_nonrenderable_cross_manufacturer_source_signature"
            if passed
            else "close_exact_velvia_revision_source_signature"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _stable_identity(report)
    encoded = canonical_json(report)
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
