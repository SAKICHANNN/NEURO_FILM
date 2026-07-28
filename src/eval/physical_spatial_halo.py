"""Frozen U6.P5B severe-halo audit for explicit spatial-response stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from src.eval.physical_spatial_response import (
    _pipeline,
    _profile,
    _slanted_edge,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    SpatialResponseProfile,
    apply_development_adjacency,
    apply_scanner_mtf,
    density_to_scan_transmittance,
)


SCHEMA = "neuro_film.u6_p5b_spatial_halo_audit_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5B contract")
    return payload


def _edge_spread(
    values: np.ndarray, distance: np.ndarray, oversample: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.floor(
        (distance.ravel() - float(np.min(distance))) * oversample
    ).astype(np.int64)
    sums = np.bincount(coordinates, weights=values.ravel().astype(np.float64))
    counts = np.bincount(coordinates)
    valid = counts > 0
    return (
        np.flatnonzero(valid).astype(np.float64) / oversample,
        sums[valid] / counts[valid],
    )


def _crossing(
    frequency: np.ndarray, mtf: np.ndarray, level: float
) -> float | None:
    below = np.flatnonzero(mtf <= level)
    if below.size == 0:
        return None
    index = int(below[0])
    if index == 0:
        return float(frequency[0])
    x0, x1 = frequency[index - 1 : index + 1]
    y0, y1 = mtf[index - 1 : index + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _bounded_edge_metrics(
    values: np.ndarray,
    distance: np.ndarray,
    *,
    pixel_pitch_um: float,
    nyquist_cycles_per_mm: float,
) -> dict[str, Any]:
    positions, esf = _edge_spread(values, distance)
    plateau = max(8, esf.size // 10)
    left = float(np.mean(esf[:plateau]))
    right = float(np.mean(esf[-plateau:]))
    lower, upper = sorted((left, right))
    span = upper - lower
    if span <= 1e-12:
        raise RuntimeError("edge has zero contrast")
    normalized = (esf - left) / (right - left)
    overshoot = max(0.0, float(np.max(normalized) - 1.0))
    undershoot = max(0.0, float(-np.min(normalized)))
    absolute_excursion = max(
        max(0.0, float(np.max(esf) - upper)),
        max(0.0, float(lower - np.min(esf))),
    )
    outside = np.maximum(lower - esf, esf - upper) > 0.01 * span
    halo_width = float(np.count_nonzero(outside)) / 4.0

    line_spread = np.gradient(esf, positions)
    spectrum = np.abs(np.fft.rfft(line_spread * np.hanning(line_spread.size)))
    if spectrum[0] <= 0.0:
        raise RuntimeError("edge MTF has zero DC")
    mtf = spectrum / spectrum[0]
    spacing = float(np.mean(np.diff(positions)))
    frequency = (
        np.fft.rfftfreq(line_spread.size, d=spacing)
        * 1000.0
        / pixel_pitch_um
    )
    keep = frequency <= nyquist_cycles_per_mm + 1e-12
    frequency = frequency[keep]
    mtf = mtf[keep]
    mtf50 = _crossing(frequency, mtf, 0.5)
    mtf10 = _crossing(frequency, mtf, 0.1)
    return {
        "plateau_left": left,
        "plateau_right": right,
        "normalized_overshoot": overshoot,
        "normalized_undershoot": undershoot,
        "absolute_scan_linear_halo_excursion": absolute_excursion,
        "halo_width_pixels_at_one_percent": halo_width,
        "mtf50_cycles_per_mm": mtf50,
        "mtf10_cycles_per_mm": mtf10,
        "mtf50_nyquist_censored": mtf50 is None,
        "mtf10_nyquist_censored": mtf10 is None,
        "maximum_evaluated_frequency_cycles_per_mm": float(frequency[-1]),
    }


def _zero_adjacency(profile: SpatialResponseProfile) -> SpatialResponseProfile:
    return SpatialResponseProfile(
        pixel_pitch_um=profile.pixel_pitch_um,
        forward_scatter_sigma_um_rgb=profile.forward_scatter_sigma_um_rgb,
        development_adjacency_sigma_um_rgb=(
            profile.development_adjacency_sigma_um_rgb
        ),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=profile.dye_diffusion_sigma_um_rgb,
        scanner_mtf_sigma_um_rgb=profile.scanner_mtf_sigma_um_rgb,
        gaussian_truncate=profile.gaussian_truncate,
    )


def _diagnostic_crop(values: np.ndarray, half_width: int) -> np.ndarray:
    height, width, _ = values.shape
    y0 = max(0, height // 2 - half_width)
    y1 = min(height, height // 2 + half_width)
    x0 = max(0, width // 2 - half_width)
    x1 = min(width, width // 2 + half_width)
    crop = np.clip(values[y0:y1, x0:x1], 0.0, 1.0)
    return np.rint(crop * 255.0).astype(np.uint8)


def _save_diagnostic(
    adjacency_outputs: list[np.ndarray],
    full_outputs: list[np.ndarray],
    baselines: list[np.ndarray],
    path: Path,
    half_width: int,
) -> str:
    tile_size = 2 * half_width
    label_height = 22
    columns = len(adjacency_outputs)
    canvas = Image.new(
        "RGB",
        (columns * tile_size, 3 * (tile_size + label_height)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    rows = (
        ("adjacency", adjacency_outputs),
        ("full", full_outputs),
        ("difference x8", None),
    )
    for row_index, (label, outputs) in enumerate(rows):
        y_label = row_index * (tile_size + label_height)
        draw.text((4, y_label + 4), label, fill=(235, 235, 235))
        for column in range(columns):
            if outputs is None:
                difference = (adjacency_outputs[column] - baselines[column]) * 8.0
                tile = np.rint(
                    np.clip(0.5 + difference, 0.0, 1.0) * 255.0
                ).astype(np.uint8)
                tile = _diagnostic_crop(tile.astype(np.float64) / 255.0, half_width)
            else:
                tile = _diagnostic_crop(outputs[column], half_width)
            image = Image.fromarray(tile, mode="RGB")
            canvas.paste(image, (column * tile_size, y_label + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_spatial_halo(
    contract: dict[str, Any],
    p5a_contract: dict[str, Any],
    sensitometry_config: dict[str, Any],
    *,
    diagnostic_path: Path,
    adjacency_apply: Callable[
        [np.ndarray, SpatialResponseProfile], np.ndarray
    ] = apply_development_adjacency,
) -> dict[str, Any]:
    profile = _profile(p5a_contract)
    zero_profile = _zero_adjacency(profile)
    operator = build_operator(sensitometry_config)
    shape = tuple(int(value) for value in contract["charts"]["shape"])
    slant = float(contract["charts"]["slant_degrees"])
    nyquist = float(contract["measurements"]["nyquist_cycles_per_mm"])
    half_width = int(contract["charts"]["diagnostic_crop_half_width_pixels"])

    density_rows: list[dict[str, Any]] = []
    adjacency_outputs: list[np.ndarray] = []
    density_baselines: list[np.ndarray] = []
    repeat_exact = True
    for index, pair in enumerate(contract["charts"]["density_edges"]):
        density, distance = _slanted_edge(
            shape, slant, float(pair[0]), float(pair[1])
        )
        baseline = density_to_scan_transmittance(density)
        output = density_to_scan_transmittance(
            adjacency_apply(density, profile)
        )
        repeat = density_to_scan_transmittance(
            adjacency_apply(density, profile)
        )
        repeat_exact = repeat_exact and np.array_equal(output, repeat)
        adjacency_outputs.append(output)
        density_baselines.append(baseline)
        density_rows.append(
            {
                "edge_index": index,
                "low_high_density": [float(pair[0]), float(pair[1])],
                "channels": [
                    _bounded_edge_metrics(
                        output[..., channel],
                        distance,
                        pixel_pitch_um=profile.pixel_pitch_um,
                        nyquist_cycles_per_mm=nyquist,
                    )
                    for channel in range(3)
                ],
            }
        )

    exposure_rows: list[dict[str, Any]] = []
    full_outputs: list[np.ndarray] = []
    for index, pair in enumerate(contract["charts"]["exposure_edges"]):
        exposure, distance = _slanted_edge(
            shape, slant, float(pair[0]), float(pair[1])
        )
        output = _pipeline(
            exposure, operator, profile, adjacency_apply=adjacency_apply
        )
        repeat = _pipeline(
            exposure, operator, profile, adjacency_apply=adjacency_apply
        )
        repeat_exact = repeat_exact and np.array_equal(output, repeat)
        full_outputs.append(output)
        exposure_rows.append(
            {
                "edge_index": index,
                "low_high_exposure": [float(pair[0]), float(pair[1])],
                "channels": [
                    _bounded_edge_metrics(
                        output[..., channel],
                        distance,
                        pixel_pitch_um=profile.pixel_pitch_um,
                        nyquist_cycles_per_mm=nyquist,
                    )
                    for channel in range(3)
                ],
            }
        )

    zero_control_exact = True
    for pair in contract["charts"]["density_edges"]:
        density, _ = _slanted_edge(shape, slant, float(pair[0]), float(pair[1]))
        zero_control_exact = zero_control_exact and np.array_equal(
            density_to_scan_transmittance(
                adjacency_apply(density, zero_profile)
            ),
            density_to_scan_transmittance(density),
        )

    probe_density, _ = _slanted_edge(shape, slant, 0.25, 1.5)
    correct = apply_scanner_mtf(
        density_to_scan_transmittance(
            adjacency_apply(probe_density, profile)
        ),
        profile,
    )
    adjacency_density = adjacency_apply(probe_density, profile)
    scale = max(float(np.max(adjacency_density)), 1.0)
    wrong = density_to_scan_transmittance(
        apply_scanner_mtf(
            adjacency_density / scale,
            profile,
        )
        * scale
    )
    wrong_order_max_abs = float(np.max(np.abs(correct - wrong)))

    diagnostic_sha = _save_diagnostic(
        adjacency_outputs,
        full_outputs,
        density_baselines,
        diagnostic_path,
        half_width,
    )
    all_metrics = [
        metric
        for row in (*density_rows, *exposure_rows)
        for metric in row["channels"]
    ]
    full_metrics = [
        metric for row in exposure_rows for metric in row["channels"]
    ]
    gates = contract["automatic_gates"]
    finite_domain = all(
        np.all(np.isfinite(output))
        and np.all(output > 0.0)
        and np.all(output <= 1.0)
        for output in (*adjacency_outputs, *full_outputs)
    )
    reported_mtf = [
        value
        for metric in all_metrics
        for value in (
            metric["mtf50_cycles_per_mm"],
            metric["mtf10_cycles_per_mm"],
        )
        if value is not None
    ]
    full_mtf50 = [
        metric["mtf50_cycles_per_mm"]
        for metric in full_metrics
        if metric["mtf50_cycles_per_mm"] is not None
    ]
    decisions = {
        "repeat": repeat_exact,
        "finite_domain": finite_domain,
        "overshoot": max(
            metric["normalized_overshoot"] for metric in all_metrics
        )
        <= float(gates["maximum_normalized_overshoot"]),
        "undershoot": max(
            metric["normalized_undershoot"] for metric in all_metrics
        )
        <= float(gates["maximum_normalized_undershoot"]),
        "absolute_halo": max(
            metric["absolute_scan_linear_halo_excursion"]
            for metric in all_metrics
        )
        <= float(gates["maximum_absolute_scan_linear_halo_excursion"]),
        "halo_width": max(
            metric["halo_width_pixels_at_one_percent"] for metric in all_metrics
        )
        <= float(gates["maximum_halo_width_pixels_at_one_percent"]),
        "nyquist_reporting": all(value <= nyquist for value in reported_mtf),
        "full_mtf50": len(full_mtf50) == len(full_metrics)
        and min(full_mtf50)
        >= float(gates["minimum_full_pipeline_mtf50_cycles_per_mm"])
        and max(full_mtf50)
        <= float(gates["maximum_full_pipeline_mtf50_cycles_per_mm"]),
        "zero_adjacency": zero_control_exact,
        "wrong_order": wrong_order_max_abs > 1e-4,
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p5b_spatial_halo_audit_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "density_edge_metrics": density_rows,
        "exposure_edge_metrics": exposure_rows,
        "diagnostic_sha256": diagnostic_sha,
        "wrong_order_max_abs": wrong_order_max_abs,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
