"""Exact raster-graph source-signature audit for Kodak VISION3 200T (2026)."""

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

from src.eval.kodak_verita_source_signature import (
    CHANNELS,
    DOMAINS,
    _compare_characteristic,
    _compare_log_domain,
    _load_vision3_curves,
    _vision3_domain_samples,
    canonical_json,
    hash_file,
)

SCHEMA = "neuro_film.u5-r2bu9-kodak-vision3-200t-raster-signature-contract.v1"
REPORT_SCHEMA = "neuro_film.u5-r2bu9-kodak-vision3-200t-raster-signature-report.v1"
STOCK = "kodak_vision3_200t_5213_7213"
VISION3_STOCKS = (
    "kodak_vision3_50d_5203_7203",
    "kodak_vision3_250d_5207_7207",
    "kodak_vision3_500t_5219_7219",
)
_LEGACY_COMPARATOR_STOCK = "kodak_verita_200d_5206_7206"


class Vision3200TRasterSignatureError(RuntimeError):
    """Raised when frozen source or extraction facts drift."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Vision3200TRasterSignatureError("path must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_gates = {
        "required_source_pages": 4,
        "required_raster_domains": 3,
        "required_channels_per_domain": 3,
        "minimum_material_channels_per_domain_pair": 2,
        "minimum_material_domains_per_vision3_pair": 2,
        "minimum_material_vision3_pairs": 3,
        "maximum_characteristic_monotonic_reversal_density": 0.015,
        "minimum_characteristic_density_span": 1.5,
        "minimum_characteristic_10_to_90_exposure_span": 2.0,
        "minimum_mtf_response_percent": 1.0,
        "maximum_mtf_response_percent": 110.0,
        "maximum_mtf_monotonic_increase_percent": 3.0,
        "minimum_granularity_sigma_d": 0.001,
        "maximum_granularity_sigma_d": 0.05,
        "two_byte_identical_audits": True,
        "visual_overlay_required": True,
    }
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU9"
        or payload.get("source", {}).get("stock_id") != STOCK
        or tuple(payload.get("comparison", {}).get("vision3_stocks", ()))
        != VISION3_STOCKS
        or payload.get("gates") != expected_gates
    ):
        raise Vision3200TRasterSignatureError("BU9 contract identity or gates drift")
    return payload


def _axis_pixel(value: float, axis: Mapping[str, Any]) -> float:
    (v0, p0), (v1, p1) = axis["anchors"]
    if axis["scale"] == "log10":
        f = (math.log10(value) - math.log10(v0)) / (
            math.log10(v1) - math.log10(v0)
        )
    else:
        f = (value - v0) / (v1 - v0)
    return float(p0 + f * (p1 - p0))


def _axis_value(pixel: float, axis: Mapping[str, Any]) -> float:
    (v0, p0), (v1, p1) = axis["anchors"]
    f = (pixel - p0) / (p1 - p0)
    if axis["scale"] == "log10":
        return float(10.0 ** (math.log10(v0) + f * (math.log10(v1) - math.log10(v0))))
    return float(v0 + f * (v1 - v0))


def _trace_dark_curve(
    gray: np.ndarray,
    *,
    x_start: int,
    x_end: int,
    y_min: int,
    y_max: int,
    endpoint_y: int,
    maximum_step: int,
    continuity_penalty: float,
) -> list[tuple[int, int]]:
    """Trace a black curve right-to-left from a frozen, label-free endpoint."""

    xs = list(range(x_end, x_start - 1, -1))
    ys = np.arange(y_min, y_max + 1, dtype=np.int32)
    cost = np.full(len(ys), np.inf, dtype=np.float64)
    cost[endpoint_y - y_min] = 0.0
    backs: list[np.ndarray] = []
    for x in xs:
        pixel_cost = gray[ys, x].astype(np.float64) / 255.0
        next_cost = np.full_like(cost, np.inf)
        back = np.full(len(ys), -1, dtype=np.int32)
        for j in range(len(ys)):
            lo = max(0, j - maximum_step)
            hi = min(len(ys), j + maximum_step + 1)
            indices = np.arange(lo, hi)
            values = cost[lo:hi] + continuity_penalty * np.abs(indices - j)
            selected = int(np.argmin(values))
            next_cost[j] = float(values[selected] + pixel_cost[j])
            back[j] = int(indices[selected])
        cost = next_cost
        backs.append(back)
    j = int(np.argmin(cost))
    reverse: list[tuple[int, int]] = []
    for index in range(len(xs) - 1, -1, -1):
        reverse.append((xs[index], int(ys[j])))
        if index:
            j = int(backs[index][j])
    return list(reversed(reverse))


def _extract_images(page: Any, config: Mapping[str, Any]) -> dict[str, Image.Image]:
    result: dict[str, Image.Image] = {}
    images = list(page.images)
    for domain in DOMAINS:
        spec = config["raster_source"][domain]
        index = int(spec["embedded_image_index"])
        if index >= len(images):
            raise Vision3200TRasterSignatureError("embedded graph index is absent")
        item = images[index]
        if hashlib.sha256(item.data).hexdigest() != spec["png_sha256"]:
            raise Vision3200TRasterSignatureError(f"{domain} embedded graph drift")
        image = Image.open(__import__("io").BytesIO(item.data)).convert("RGB")
        if list(image.size) != spec["dimensions"]:
            raise Vision3200TRasterSignatureError(f"{domain} graph dimensions drift")
        result[domain] = image
    return result


def _trace_domain(
    image: Image.Image,
    spec: Mapping[str, Any],
    coordinates: Sequence[float],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[tuple[int, int]]]]:
    gray = np.asarray(image.convert("L"))
    traces: dict[str, list[tuple[int, int]]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    x_pixels = [_axis_pixel(float(value), spec["x_axis"]) for value in coordinates]
    x_start = math.floor(min(x_pixels))
    x_end = math.ceil(max(x_pixels))
    for channel in CHANNELS:
        trace = _trace_dark_curve(
            gray,
            x_start=x_start,
            x_end=x_end,
            y_min=int(spec["trace_y_bounds"][0]),
            y_max=int(spec["trace_y_bounds"][1]),
            endpoint_y=int(spec["endpoint_y"][channel]),
            maximum_step=int(spec["maximum_step"]),
            continuity_penalty=float(spec["continuity_penalty"]),
        )
        traces[channel] = trace
        lookup = {x: y for x, y in trace}
        sampled = [
            _axis_value(float(lookup[round(pixel)]), spec["y_axis"])
            for pixel in x_pixels
        ]
        values[channel] = {
            "coordinates": [float(value) for value in coordinates],
            "values": sampled,
        }
    return values, traces


def _write_overlay(path: Path, images: Mapping[str, Image.Image], traces: Mapping[str, Any]) -> str:
    colors = {"blue": (0, 100, 255), "green": (0, 180, 0), "red": (255, 0, 0)}
    panels: list[Image.Image] = []
    for domain in DOMAINS:
        panel = images[domain].copy()
        draw = ImageDraw.Draw(panel)
        for channel in CHANNELS:
            draw.line(traces[domain][channel], fill=colors[channel], width=1)
        panels.append(panel)
    canvas = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hash_file(path)


def _rename_comparator_pairs(rows: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key.replace(_LEGACY_COMPARATOR_STOCK, STOCK, 1): value
        for key, value in rows.items()
    }


def _rename_pair_ids(rows: Sequence[str]) -> list[str]:
    return [value.replace(_LEGACY_COMPARATOR_STOCK, STOCK, 1) for value in rows]


def audit_source_signature(config: Mapping[str, Any], root: Path, *, overlay_path: Path) -> dict[str, Any]:
    source = config["source"]
    source_path = root / _relative_path(source["path"])
    if not source_path.is_file() or source_path.stat().st_size != source["expected_bytes"] or hash_file(source_path) != source["sha256"]:
        raise Vision3200TRasterSignatureError("BU9 source integrity mismatch")
    reader = PdfReader(source_path)
    if len(reader.pages) != config["gates"]["required_source_pages"]:
        raise Vision3200TRasterSignatureError("BU9 page count drift")
    normalized_text = " ".join("\n".join(page.extract_text() or "" for page in reader.pages).split())
    text_gate = all(" ".join(anchor.split()) in normalized_text for anchor in config["required_text_anchors"])
    page = reader.pages[config["raster_source"]["page_zero_based"]]
    images = _extract_images(page, config)
    coordinates = {
        "characteristic": config["comparison"]["characteristic"]["sample_log_relative_exposures"],
        "mtf": config["comparison"]["mtf"]["common_frequencies_cycles_per_mm"],
        "granularity": config["comparison"]["granularity"]["common_log_relative_exposures"],
    }
    observed: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    for domain in DOMAINS:
        observed[domain], traces[domain] = _trace_domain(images[domain], config["raster_source"][domain], coordinates[domain])
    parents, parent_hashes = _load_vision3_curves(root, config)
    vision3 = {
        "characteristic": _vision3_domain_samples(parents["characteristic_trace"], coordinates["characteristic"], domain="characteristic"),
        "mtf": _vision3_domain_samples(parents["mtf_trace"], coordinates["mtf"], domain="mtf"),
        "granularity": _vision3_domain_samples(parents["granularity_trace"], coordinates["granularity"], domain="granularity"),
    }
    characteristic_pairwise, characteristic_summary = _compare_characteristic(config, observed["characteristic"], vision3["characteristic"])
    mtf_pairwise, mtf_material = _compare_log_domain(config, observed["mtf"], vision3["mtf"], domain="mtf")
    gran_pairwise, gran_material = _compare_log_domain(config, observed["granularity"], vision3["granularity"], domain="granularity")
    characteristic_pairwise = _rename_comparator_pairs(characteristic_pairwise)
    mtf_pairwise = _rename_comparator_pairs(mtf_pairwise)
    gran_pairwise = _rename_comparator_pairs(gran_pairwise)
    characteristic_summary["material_pairs"] = _rename_pair_ids(characteristic_summary["material_pairs"])
    mtf_material = _rename_pair_ids(mtf_material)
    gran_material = _rename_pair_ids(gran_material)
    pairwise = {"characteristic": characteristic_pairwise, "mtf": mtf_pairwise, "granularity": gran_pairwise}
    material_domains_by_pair: dict[str, list[str]] = {}
    material_stock_pairs: list[str] = []
    for stock in VISION3_STOCKS:
        pair_id = f"{STOCK}__vs__{stock}"
        domains = [domain for domain in DOMAINS if pairwise[domain][pair_id]["material_domain_pair"]]
        material_domains_by_pair[pair_id] = domains
        if len(domains) >= config["gates"]["minimum_material_domains_per_vision3_pair"]:
            material_stock_pairs.append(pair_id)
    characteristic_gate = all(
        diagnostics["maximum_monotonic_reversal_density"] <= config["gates"]["maximum_characteristic_monotonic_reversal_density"]
        and diagnostics["density_span"] >= config["gates"]["minimum_characteristic_density_span"]
        and diagnostics["x10_to_x90_span"] >= config["gates"]["minimum_characteristic_10_to_90_exposure_span"]
        for diagnostics in characteristic_summary["verita"].values()
    )
    mtf_values = np.asarray([v for c in CHANNELS for v in observed["mtf"][c]["values"]])
    mtf_increases = {c: float(np.max(np.maximum(0.0, np.diff(observed["mtf"][c]["values"]) / np.asarray(observed["mtf"][c]["values"][:-1]) * 100.0))) for c in CHANNELS}
    mtf_gate = bool(np.all(np.isfinite(mtf_values)) and np.min(mtf_values) >= config["gates"]["minimum_mtf_response_percent"] and np.max(mtf_values) <= config["gates"]["maximum_mtf_response_percent"] and max(mtf_increases.values()) <= config["gates"]["maximum_mtf_monotonic_increase_percent"])
    gran_values = np.asarray([v for c in CHANNELS for v in observed["granularity"][c]["values"]])
    gran_gate = bool(np.all(np.isfinite(gran_values)) and np.min(gran_values) >= config["gates"]["minimum_granularity_sigma_d"] and np.max(gran_values) <= config["gates"]["maximum_granularity_sigma_d"])
    overlay_sha = _write_overlay(overlay_path, images, traces)
    gate_results = {
        "source_integrity": True,
        "first_party_text_anchors": text_gate,
        "exact_embedded_raster_graphs": True,
        "three_domains_three_channels": len(observed) == 3 and all(len(observed[d]) == 3 for d in DOMAINS),
        "vision3_parent_evidence_exact": len(parent_hashes) == len(config["vision3_evidence"]),
        "characteristic_source_range": characteristic_gate,
        "mtf_source_range_and_monotonicity": mtf_gate,
        "granularity_source_range": gran_gate,
        "multi_domain_stock_signature": len(material_stock_pairs) >= config["gates"]["minimum_material_vision3_pairs"],
        "visual_overlay_generated": bool(overlay_sha),
        "no_operator_fit_or_photographic_pixels": True,
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "source": {"pdf_sha256": source["sha256"], "pdf_bytes": source["expected_bytes"], "pdf_pages": len(reader.pages), "stock_id": STOCK, "measurement_context": source["measurement_context"]},
        "embedded_graph_sha256": {d: config["raster_source"][d]["png_sha256"] for d in DOMAINS},
        "parent_evidence_sha256": parent_hashes,
        "observed_samples": observed,
        "characteristic_diagnostics": characteristic_summary["verita"],
        "mtf_maximum_monotonic_increase_percent": mtf_increases,
        "pairwise": pairwise,
        "material_pairs_by_domain": {"characteristic": characteristic_summary["material_pairs"], "mtf": mtf_material, "granularity": gran_material},
        "material_domains_by_pair": material_domains_by_pair,
        "material_stock_pairs": material_stock_pairs,
        "overlay_sha256": overlay_sha,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    return {"schema": REPORT_SCHEMA, **stable, "signature_pass": passed, "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(), "decision": config["branch_rule"]["pass" if passed else "fail"], "claim_ceiling": config["claim_ceiling"]}


__all__ = ["REPORT_SCHEMA", "SCHEMA", "STOCK", "Vision3200TRasterSignatureError", "audit_source_signature", "load_contract"]
