"""Synthetic identifiability audit for explicit U6.P5A spatial stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    SpatialResponseProfile,
    apply_development_adjacency,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
    density_to_scan_transmittance,
)


SCHEMA = "neuro_film.u6_p5a_spatial_response_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5A contract")
    return payload


def _profile(contract: dict[str, Any], *, zero: bool = False) -> SpatialResponseProfile:
    row = contract["profile"]
    zeros = (0.0, 0.0, 0.0)
    return SpatialResponseProfile(
        pixel_pitch_um=float(contract["physical_scale"]["pixel_pitch_um"]),
        forward_scatter_sigma_um_rgb=(
            zeros
            if zero
            else tuple(float(value) for value in row["forward_scatter_sigma_um_rgb"])
        ),
        development_adjacency_sigma_um_rgb=(
            zeros
            if zero
            else tuple(
                float(value)
                for value in row["development_adjacency_sigma_um_rgb"]
            )
        ),
        development_adjacency_gain_rgb=(
            zeros
            if zero
            else tuple(float(value) for value in row["development_adjacency_gain_rgb"])
        ),
        dye_diffusion_sigma_um_rgb=(
            zeros
            if zero
            else tuple(float(value) for value in row["dye_diffusion_sigma_um_rgb"])
        ),
        scanner_mtf_sigma_um_rgb=(
            zeros
            if zero
            else tuple(float(value) for value in row["scanner_mtf_sigma_um_rgb"])
        ),
        gaussian_truncate=float(row["gaussian_truncate"]),
    )


def _slanted_edge(
    shape: tuple[int, int],
    slant_degrees: float,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    distance = (
        x
        - (width - 1) / 2.0
        - math.tan(math.radians(slant_degrees)) * (y - (height - 1) / 2.0)
    )
    values = np.where(distance < 0.0, low, high).astype(np.float64)
    return np.repeat(values[..., None], 3, axis=-1), distance


def _edge_spread(
    values: np.ndarray, distance: np.ndarray, oversample: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.floor(
        (distance.ravel() - float(np.min(distance))) * oversample
    ).astype(np.int64)
    sums = np.bincount(
        coordinates, weights=values.ravel().astype(np.float64)
    )
    counts = np.bincount(coordinates)
    valid = counts > 0
    esf = sums[valid] / counts[valid]
    positions = np.flatnonzero(valid).astype(np.float64) / oversample
    return positions, esf


def _crossing(frequency: np.ndarray, mtf: np.ndarray, level: float) -> float:
    below = np.flatnonzero(mtf <= level)
    if below.size == 0:
        return float(frequency[-1])
    index = int(below[0])
    if index == 0:
        return float(frequency[0])
    x0, x1 = frequency[index - 1 : index + 1]
    y0, y1 = mtf[index - 1 : index + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _edge_metrics(
    values: np.ndarray,
    distance: np.ndarray,
    *,
    pixel_pitch_um: float,
) -> dict[str, float]:
    positions, esf = _edge_spread(values, distance)
    plateau = max(8, esf.size // 10)
    left = float(np.mean(esf[:plateau]))
    right = float(np.mean(esf[-plateau:]))
    span = right - left
    if abs(span) <= 1e-12:
        raise RuntimeError("edge has zero contrast")
    normalized = (esf - left) / span
    overshoot = max(0.0, float(np.max(normalized) - 1.0))
    undershoot = max(0.0, float(-np.min(normalized)))
    line_spread = np.gradient(esf, positions)
    spectrum = np.abs(np.fft.rfft(line_spread * np.hanning(line_spread.size)))
    if spectrum[0] <= 0.0:
        raise RuntimeError("edge MTF has zero DC")
    mtf = spectrum / spectrum[0]
    spacing = float(np.mean(np.diff(positions)))
    frequency_cpp = np.fft.rfftfreq(line_spread.size, d=spacing)
    scale = 1000.0 / pixel_pitch_um
    return {
        "mtf50_cycles_per_mm": _crossing(frequency_cpp, mtf, 0.5) * scale,
        "mtf10_cycles_per_mm": _crossing(frequency_cpp, mtf, 0.1) * scale,
        "overshoot": overshoot,
        "undershoot": undershoot,
    }


def _pipeline(
    exposure: np.ndarray,
    operator: Any,
    profile: SpatialResponseProfile,
    *,
    forward: bool = True,
    adjacency: bool = True,
    diffusion: bool = True,
    scanner: bool = True,
) -> np.ndarray:
    layer_exposure = (
        apply_forward_scatter(exposure, profile) if forward else exposure
    )
    density = operator.apply(layer_exposure)
    if adjacency:
        density = apply_development_adjacency(density, profile)
    if diffusion:
        density = apply_dye_diffusion(density, profile)
    scan = density_to_scan_transmittance(density)
    if scanner:
        scan = apply_scanner_mtf(scan, profile)
    return scan


def evaluate_spatial_response(
    contract: dict[str, Any], sensitometry_config: dict[str, Any]
) -> dict[str, Any]:
    operator = build_operator(sensitometry_config)
    profile = _profile(contract)
    shape = tuple(int(value) for value in contract["charts"]["slanted_edge_shape"])
    slant = float(contract["charts"]["slant_degrees"])
    pitch = profile.pixel_pitch_um
    edge_rows = []
    repeat_outputs = []
    for pair_index, pair in enumerate(contract["charts"]["exposure_pairs"]):
        exposure, distance = _slanted_edge(shape, slant, float(pair[0]), float(pair[1]))
        output = _pipeline(exposure, operator, profile)
        repeat_outputs.append(
            np.array_equal(output, _pipeline(exposure, operator, profile))
        )
        edge_rows.append(
            {
                "pair_index": pair_index,
                "low_high": [float(pair[0]), float(pair[1])],
                "channels": [
                    _edge_metrics(
                        output[..., channel], distance, pixel_pitch_um=pitch
                    )
                    for channel in range(3)
                ],
            }
        )
    exposure, distance = _slanted_edge(shape, slant, 0.03, 1.0)
    full = _pipeline(exposure, operator, profile)
    ablations = {
        "forward": float(
            np.max(np.abs(full - _pipeline(exposure, operator, profile, forward=False)))
        ),
        "adjacency": float(
            np.max(np.abs(full - _pipeline(exposure, operator, profile, adjacency=False)))
        ),
        "diffusion": float(
            np.max(np.abs(full - _pipeline(exposure, operator, profile, diffusion=False)))
        ),
        "scanner": float(
            np.max(np.abs(full - _pipeline(exposure, operator, profile, scanner=False)))
        ),
    }
    density = apply_dye_diffusion(
        apply_development_adjacency(
            operator.apply(apply_forward_scatter(exposure, profile)), profile
        ),
        profile,
    )
    density_scale = max(float(np.max(density)), 1.0)
    wrong_density = (
        apply_scanner_mtf(density / density_scale, profile) * density_scale
    )
    wrong_order = density_to_scan_transmittance(wrong_density)
    correct_vs_wrong = float(np.max(np.abs(full - wrong_order)))
    density_edge, density_distance = _slanted_edge(shape, slant, 0.5, 1.5)
    dye_scan = density_to_scan_transmittance(
        apply_dye_diffusion(density_edge, profile)
    )
    dye_mtf50 = [
        _edge_metrics(dye_scan[..., channel], density_distance, pixel_pitch_um=pitch)[
            "mtf50_cycles_per_mm"
        ]
        for channel in range(3)
    ]
    scan_edge, scan_distance = _slanted_edge(shape, slant, 0.1, 0.8)
    scanner_only = apply_scanner_mtf(scan_edge, profile)
    scanner_mtf50 = [
        _edge_metrics(
            scanner_only[..., channel], scan_distance, pixel_pitch_um=pitch
        )["mtf50_cycles_per_mm"]
        for channel in range(3)
    ]
    adjacency_scan = density_to_scan_transmittance(
        apply_development_adjacency(density_edge, profile)
    )
    adjacency_metrics = [
        _edge_metrics(
            adjacency_scan[..., channel], density_distance, pixel_pitch_um=pitch
        )
        for channel in range(3)
    ]
    height, width = shape
    x = np.arange(width, dtype=np.float64)
    line_modulations = []
    for frequency_mm in contract["charts"]["line_pair_cycles_per_mm"]:
        frequency_cpp = float(frequency_mm) * pitch / 1000.0
        wave = np.where(
            np.sin(2.0 * np.pi * frequency_cpp * x) >= 0.0, 1.0, 0.03
        )
        chart = np.broadcast_to(wave[None, :, None], (height, width, 3)).copy()
        result = _pipeline(chart, operator, profile)
        central = result[height // 4 : 3 * height // 4]
        channels = []
        for channel in range(3):
            low = float(np.min(central[..., channel]))
            high = float(np.max(central[..., channel]))
            channels.append((high - low) / (high + low))
        line_modulations.append(
            {"cycles_per_mm": float(frequency_mm), "channels": channels}
        )
    zero_output = _pipeline(exposure, operator, _profile(contract, zero=True))
    identity_reference = density_to_scan_transmittance(operator.apply(exposure))
    identity_exact = np.array_equal(zero_output, identity_reference)
    gates = contract["automatic_gates"]
    forward_mtf50 = [
        row["channels"][1]["mtf50_cycles_per_mm"] for row in edge_rows
    ]
    dye_margin = min(
        dye_mtf50[0] - dye_mtf50[1],
        dye_mtf50[1] - dye_mtf50[2],
    )
    adjacency_overshoot = max(
        item["overshoot"] for item in adjacency_metrics
    )
    all_edge_metrics = [
        metric for row in edge_rows for metric in row["channels"]
    ]
    decisions = {
        "repeat": all(repeat_outputs),
        "domain": bool(
            np.all(np.isfinite(full))
            and np.all(full > 0.0)
            and np.all(full <= 1.0)
        ),
        "ablation": min(ablations.values())
        >= float(gates["minimum_stage_ablation_max_abs"]),
        "order": correct_vs_wrong
        >= float(gates["minimum_correct_vs_wrong_order_max_abs"]),
        "forward_exposure": max(forward_mtf50) - min(forward_mtf50)
        >= float(gates["minimum_forward_mtf50_exposure_spread_cycles_per_mm"]),
        "adjacency": adjacency_overshoot
        >= float(gates["adjacency_overshoot_min"])
        and adjacency_overshoot <= float(gates["adjacency_overshoot_max"]),
        "dye_channels": dye_margin
        >= float(gates["dye_mtf50_channel_margin_cycles_per_mm"]),
        "scanner_neutral": max(scanner_mtf50) - min(scanner_mtf50)
        <= float(gates["scanner_neutral_mtf50_channel_spread_max_cycles_per_mm"]),
        "mtf_order": all(
            item["mtf10_cycles_per_mm"] > item["mtf50_cycles_per_mm"]
            for item in all_edge_metrics
        ),
        "line_pairs": min(
            value for row in line_modulations for value in row["channels"]
        )
        >= float(gates["line_pair_modulation_min"]),
        "identity": identity_exact,
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p5a_spatial_response_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "edge_metrics": edge_rows,
        "stage_ablation_max_abs": ablations,
        "correct_vs_wrong_order_max_abs": correct_vs_wrong,
        "dye_mtf50_cycles_per_mm": dye_mtf50,
        "scanner_neutral_mtf50_cycles_per_mm": scanner_mtf50,
        "adjacency_edge_metrics": adjacency_metrics,
        "line_pair_modulation": line_modulations,
        "identity_exact": identity_exact,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
        "negative_controls": contract["negative_controls"],
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
