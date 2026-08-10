"""U5.R2BY0 Kodak 2242/3242/5242 intermediate-material source audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader

from src.eval.kodak_verita_source_signature import (
    _axis_value,
    _vision3_domain_samples,
    canonical_json,
    hash_file,
)

SCHEMA = "neuro_film.u5_r2by0_kodak_2242_intermediate_role_signature_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2by0_kodak_2242_intermediate_role_signature_report.v1"
)
EXPERIMENT_ID = "U5.R2BY0"
MATERIAL = "kodak_vision_color_intermediate_5242_2242_3242"
VISION3_STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
CHANNELS = ("blue", "green", "red")
DOMAINS = ("mtf", "granularity")


class IntermediateRoleSignatureError(RuntimeError):
    """Raised when a frozen source, trace, parent, or contract drifts."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise IntermediateRoleSignatureError("BY0 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("source", {}).get("material_id") != MATERIAL
        or payload.get("source", {}).get("sha256")
        != "85404f73abb698567e7ad6b84f621f724b99b6e7159c864c14f725c5ab40be63"
        or tuple(comparison.get("vision3_stocks", ())) != VISION3_STOCKS
        or tuple(comparison.get("channels", ())) != CHANNELS
        or tuple(comparison.get("domains", ())) != DOMAINS
        or float(comparison.get("material_threshold", -1.0)) != 0.08
        or int(comparison.get("trace_uncertainty_pixels", -1)) != 2
        or gates.get("minimum_material_channels_per_domain_pair") != 2
        or gates.get("minimum_material_domains_per_vision3_pair") != 2
        or gates.get("minimum_material_vision3_pairs") != 3
        or gates.get("require_material_classification_under_trace_uncertainty")
        is not True
        or gates.get("two_byte_identical_audits") is not True
        or gates.get("visual_overlay_required") is not True
    ):
        raise IntermediateRoleSignatureError("BY0 frozen contract drift")
    _relative_path(str(payload["source"]["path"]))
    for row in payload.get("vision3_evidence", {}).values():
        _relative_path(str(row.get("path", "")))
    raster = payload.get("raster_sources", {})
    for domain in DOMAINS:
        samples = raster.get(domain, {}).get("samples", {})
        coordinates_key = (
            "frequencies_cycles_per_mm"
            if domain == "mtf"
            else "log_relative_exposures"
        )
        coordinates = samples.get(coordinates_key, ())
        x_pixels = samples.get("x_pixels", ())
        channels = samples.get("channels", {})
        if (
            len(coordinates) != len(x_pixels)
            or tuple(channels) != CHANNELS
            or any(len(channels[channel]) != len(coordinates) for channel in CHANNELS)
        ):
            raise IntermediateRoleSignatureError(f"BY0 {domain} trace shape drift")
    return payload


def _load_parents(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, row in config["vision3_evidence"].items():
        path = root / _relative_path(str(row["path"]))
        if not path.is_file() or hash_file(path) != row["sha256"]:
            raise IntermediateRoleSignatureError(f"BY0 parent evidence drift: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            "required_decision" in row
            and payload.get("decision") != row["required_decision"]
        ):
            raise IntermediateRoleSignatureError(f"BY0 parent decision drift: {name}")
        payloads[name] = payload
        hashes[name] = row["sha256"]
    return payloads, hashes


def _extract_rgb_image(page: Any, config: Mapping[str, Any]) -> np.ndarray:
    name = str(config["xobject"])
    xobjects = page["/Resources"].get("/XObject", {})
    if name not in xobjects:
        raise IntermediateRoleSignatureError(f"BY0 missing source graph {name}")
    image = xobjects[name].get_object()
    raw = image.get_data()
    width = int(config["width"])
    height = int(config["height"])
    if (
        int(image.get("/Width", -1)) != width
        or int(image.get("/Height", -1)) != height
        or str(image.get("/ColorSpace")) != "/DeviceRGB"
        or int(image.get("/BitsPerComponent", -1)) != 8
        or len(raw) != int(config["decoded_rgb_bytes"])
        or hashlib.sha256(raw).hexdigest() != config["decoded_rgb_sha256"]
    ):
        raise IntermediateRoleSignatureError("BY0 source graph identity drift")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _trace_values(
    image: np.ndarray, config: Mapping[str, Any], gates: Mapping[str, Any]
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    samples = config["samples"]
    coordinate_key = (
        "frequencies_cycles_per_mm"
        if "frequencies_cycles_per_mm" in samples
        else "log_relative_exposures"
    )
    coordinates = [float(value) for value in samples[coordinate_key]]
    x_pixels = [int(value) for value in samples["x_pixels"]]
    radius = int(gates["trace_local_radius_pixels"])
    maximum_local = int(gates["maximum_trace_local_minimum_uint8"])
    grayscale = np.min(image, axis=2)
    traces: dict[str, dict[str, list[float]]] = {}
    local_minima: dict[str, list[int]] = {}
    for channel in CHANNELS:
        y_pixels = [int(value) for value in samples["channels"][channel]]
        minima: list[int] = []
        values: list[float] = []
        for x_pixel, y_pixel in zip(x_pixels, y_pixels):
            if not (
                radius <= x_pixel < image.shape[1] - radius
                and radius <= y_pixel < image.shape[0] - radius
            ):
                raise IntermediateRoleSignatureError("BY0 trace point outside image")
            local = grayscale[
                y_pixel - radius : y_pixel + radius + 1,
                x_pixel - radius : x_pixel + radius + 1,
            ]
            local_minimum = int(np.min(local))
            if local_minimum > maximum_local:
                raise IntermediateRoleSignatureError("BY0 trace point left graph ink")
            minima.append(local_minimum)
            values.append(_axis_value(float(y_pixel), config["y_axis"]))
        traces[channel] = {"coordinates": coordinates, "values": values}
        local_minima[channel] = minima
    return traces, {"local_minimum_uint8": local_minima}


def _source_diagnostics(
    config: Mapping[str, Any], traces: Mapping[str, Any], *, domain: str
) -> dict[str, Any]:
    values = np.asarray(
        [value for channel in CHANNELS for value in traces[channel]["values"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise IntermediateRoleSignatureError(f"BY0 nonfinite {domain} trace")
    gates = config["gates"]
    if domain == "mtf":
        increases = {
            channel: float(
                np.max(
                    np.maximum(
                        0.0,
                        np.diff(np.asarray(traces[channel]["values"], dtype=np.float64))
                        / np.asarray(traces[channel]["values"][:-1], dtype=np.float64)
                        * 100.0,
                    )
                )
            )
            for channel in CHANNELS
        }
        passed = bool(
            float(np.min(values)) >= float(gates["minimum_mtf_response_percent"])
            and float(np.max(values)) <= float(gates["maximum_mtf_response_percent"])
            and max(increases.values())
            <= float(gates["maximum_mtf_monotonic_increase_percent"])
        )
        return {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "maximum_monotonic_increase_percent_by_channel": increases,
            "passed": passed,
        }
    passed = bool(
        float(np.min(values)) >= float(gates["minimum_granularity_sigma_d"])
        and float(np.max(values)) <= float(gates["maximum_granularity_sigma_d"])
    )
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "passed": passed,
    }


def _trace_log_uncertainty(config: Mapping[str, Any], pixels: int) -> float:
    axis = config["y_axis"]
    values = []
    for y_pixel in config["samples"]["channels"]["blue"]:
        center = math.log(_axis_value(float(y_pixel), axis))
        values.extend(
            abs(math.log(_axis_value(float(y_pixel + delta), axis)) - center)
            for delta in (-pixels, pixels)
        )
    return float(max(values))


def _compare_domain(
    config: Mapping[str, Any],
    material: Mapping[str, Any],
    vision3: Mapping[str, Any],
    *,
    domain: str,
) -> tuple[dict[str, Any], list[str]]:
    threshold = float(config["comparison"]["material_threshold"])
    uncertainty = _trace_log_uncertainty(
        config["raster_sources"][domain],
        int(config["comparison"]["trace_uncertainty_pixels"]),
    )
    pairwise: dict[str, Any] = {}
    material_pairs: list[str] = []
    for stock in VISION3_STOCKS:
        channels: dict[str, Any] = {}
        material_count = 0
        for channel in CHANNELS:
            first = np.log(np.asarray(material[channel]["values"], dtype=np.float64))
            second = np.log(
                np.asarray(vision3[stock][channel]["values"], dtype=np.float64)
            )
            first = first - float(np.mean(first))
            second = second - float(np.mean(second))
            rmse = float(np.sqrt(np.mean(np.square(first - second))))
            conservative = max(0.0, rmse - uncertainty)
            is_material = conservative >= threshold
            material_count += int(is_material)
            channels[channel] = {
                "mean_centered_log_rmse": rmse,
                "trace_uncertainty_log_rmse_bound": uncertainty,
                "conservative_lower_bound": conservative,
                "material": is_material,
            }
        pair_id = f"{MATERIAL}__vs__{stock}"
        pair_material = material_count >= int(
            config["gates"]["minimum_material_channels_per_domain_pair"]
        )
        if pair_material:
            material_pairs.append(pair_id)
        pairwise[pair_id] = {
            "channels": channels,
            "material_channel_count": material_count,
            "material_domain_pair": pair_material,
        }
    return pairwise, material_pairs


def _draw_overlay(
    output: Path,
    images: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> str:
    rendered: list[Image.Image] = []
    colours = {"blue": "#1465ff", "green": "#00a650", "red": "#e02020"}
    for domain in DOMAINS:
        image = Image.fromarray(images[domain], mode="RGB")
        draw = ImageDraw.Draw(image)
        samples = config["raster_sources"][domain]["samples"]
        for channel in CHANNELS:
            for x_pixel, y_pixel in zip(
                samples["x_pixels"], samples["channels"][channel]
            ):
                x = int(x_pixel)
                y = int(y_pixel)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colours[channel], width=2)
        rendered.append(image)
    width = max(image.width for image in rendered)
    height = sum(image.height for image in rendered)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in rendered:
        canvas.paste(image, (0, offset))
        offset += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return hash_file(output)


def audit_intermediate_role_signature(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> dict[str, Any]:
    source = config["source"]
    source_path = root / _relative_path(str(source["path"]))
    if (
        not source_path.is_file()
        or source_path.stat().st_size != int(source["expected_bytes"])
        or hash_file(source_path) != source["sha256"]
    ):
        raise IntermediateRoleSignatureError("BY0 source integrity mismatch")
    reader = PdfReader(str(source_path))
    if len(reader.pages) != int(config["gates"]["required_source_pages"]):
        raise IntermediateRoleSignatureError("BY0 source page count drift")
    text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
    text_gate = all(" ".join(anchor.split()) in text for anchor in config["required_text_anchors"])
    page = reader.pages[int(config["raster_sources"]["page_zero_based"])]
    images: dict[str, np.ndarray] = {}
    traces: dict[str, Any] = {}
    trace_checks: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for domain in DOMAINS:
        images[domain] = _extract_rgb_image(page, config["raster_sources"][domain])
        traces[domain], trace_checks[domain] = _trace_values(
            images[domain], config["raster_sources"][domain], config["gates"]
        )
        diagnostics[domain] = _source_diagnostics(config, traces[domain], domain=domain)
    parents, parent_hashes = _load_parents(root, config)
    vision3 = {
        "mtf": _vision3_domain_samples(
            parents["mtf_trace"],
            traces["mtf"]["blue"]["coordinates"],
            domain="mtf",
        ),
        "granularity": _vision3_domain_samples(
            parents["granularity_trace"],
            traces["granularity"]["blue"]["coordinates"],
            domain="granularity",
        ),
    }
    pairwise: dict[str, Any] = {}
    material_domains: dict[str, list[str]] = {
        f"{MATERIAL}__vs__{stock}": [] for stock in VISION3_STOCKS
    }
    for domain in DOMAINS:
        domain_rows, material_pairs = _compare_domain(
            config, traces[domain], vision3[domain], domain=domain
        )
        pairwise[domain] = domain_rows
        for pair_id in material_pairs:
            material_domains[pair_id].append(domain)
    material_stock_pairs = [
        pair_id
        for pair_id, domains in material_domains.items()
        if len(domains)
        >= int(config["gates"]["minimum_material_domains_per_vision3_pair"])
    ]
    overlay_sha256 = _draw_overlay(overlay_path, images, config)
    passed = bool(
        text_gate
        and all(row["passed"] for row in diagnostics.values())
        and len(material_stock_pairs)
        >= int(config["gates"]["minimum_material_vision3_pairs"])
    )
    decision = (
        "retain_non_renderable_intermediate_material_role_profile"
        if passed
        else "close_quantitative_intermediate_role_profile_expansion"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source": {
            "path": str(source["path"]),
            "bytes": int(source["expected_bytes"]),
            "sha256": str(source["sha256"]),
            "pages": len(reader.pages),
            "text_anchors_passed": text_gate,
        },
        "parent_hashes": parent_hashes,
        "traces": traces,
        "trace_checks": trace_checks,
        "source_diagnostics": diagnostics,
        "pairwise": pairwise,
        "material_domains_by_stock_pair": material_domains,
        "material_stock_pairs": material_stock_pairs,
        "overlay_sha256": overlay_sha256,
        "passed": passed,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(report))


__all__ = [
    "IntermediateRoleSignatureError",
    "audit_intermediate_role_signature",
    "load_contract",
    "write_report",
]
