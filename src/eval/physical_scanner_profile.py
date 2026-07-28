"""Synthetic identifiability audit for the explicit U6.P6A scanner boundary."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_spatial_halo import _bounded_edge_metrics
from src.eval.physical_spatial_response import _slanted_edge
from src.film_physics import (
    SCANNER_STAGES,
    ScannerProfile,
    apply_scanner_profile,
)


SCHEMA = "neuro_film.u6_p6a_scanner_profile_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6A contract")
    return payload


def _profile(row: dict[str, Any]) -> ScannerProfile:
    dmax = row["dmax_density_rgb"]
    return ScannerProfile(
        profile_id=row["profile_id"],
        illuminant_rgb=tuple(float(value) for value in row["illuminant_rgb"]),
        spectral_matrix=tuple(
            tuple(float(value) for value in matrix_row)
            for matrix_row in row["spectral_matrix"]
        ),
        local_flare_fraction=float(row["local_flare_fraction"]),
        global_flare_fraction=float(row["global_flare_fraction"]),
        flare_sigma_um=float(row["flare_sigma_um"]),
        dmax_density_rgb=(
            None if dmax is None else tuple(float(value) for value in dmax)
        ),
        mtf_sigma_um_rgb=tuple(
            float(value) for value in row["mtf_sigma_um_rgb"]
        ),
        shot_noise_variance_scale=float(row["shot_noise_variance_scale"]),
        read_noise_variance=float(row["read_noise_variance"]),
        seed=int(row["seed"]),
    )


def _density_wedge(shape: tuple[int, int], densities: list[float]) -> np.ndarray:
    height, width = shape
    boundaries = np.linspace(0, width, len(densities) + 1, dtype=np.int64)
    values = np.empty((height, width, 3), dtype=np.float64)
    for index, density in enumerate(densities):
        values[:, boundaries[index] : boundaries[index + 1], :] = 10.0 ** (
            -float(density)
        )
    return values


def evaluate_scanner_profiles(contract: dict[str, Any]) -> dict[str, Any]:
    profiles = {
        name: _profile(row) for name, row in contract["profiles"].items()
    }
    charts = contract["synthetic_charts"]
    shape = tuple(int(value) for value in charts["shape"])
    pitch = float(charts["pixel_pitch_um"])
    wedge = _density_wedge(shape, charts["density_wedge"])
    edge, distance = _slanted_edge(
        shape,
        float(charts["slant_degrees"]),
        float(charts["slanted_edge_low_high_transmittance"][0]),
        float(charts["slanted_edge_low_high_transmittance"][1]),
    )

    identity = apply_scanner_profile(wedge, profiles["identity"], pixel_pitch_um=pitch)
    identity_exact = np.array_equal(identity, wedge)
    outputs = {
        name: apply_scanner_profile(wedge, profile, pixel_pitch_um=pitch)
        for name, profile in profiles.items()
        if name != "identity"
    }
    repeat_exact = all(
        np.array_equal(
            output,
            apply_scanner_profile(wedge, profiles[name], pixel_pitch_um=pitch),
        )
        for name, output in outputs.items()
    )
    scanner_a = outputs["scanner_a"]
    scanner_b = outputs["scanner_b"]
    stage_ablation: dict[str, float] = {}
    for stage in SCANNER_STAGES:
        selected = tuple(item for item in SCANNER_STAGES if item != stage)
        ablated = apply_scanner_profile(
            wedge,
            profiles["scanner_a"],
            pixel_pitch_um=pitch,
            stages=selected,
        )
        stage_ablation[stage] = float(np.max(np.abs(scanner_a - ablated)))

    seeded = apply_scanner_profile(
        wedge,
        replace(profiles["scanner_a"], seed=profiles["scanner_a"].seed + 100),
        pixel_pitch_um=pitch,
    )
    seed_change = float(np.max(np.abs(scanner_a - seeded)))
    no_noise = apply_scanner_profile(
        wedge,
        profiles["scanner_a"],
        pixel_pitch_um=pitch,
        stages=SCANNER_STAGES[:-1],
    )
    new_boundary = (
        ((scanner_a <= 0.0) | (scanner_a >= 1.0))
        & ~((no_noise <= 0.0) | (no_noise >= 1.0))
    )
    new_boundary_fraction = float(np.mean(new_boundary))

    edge_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("scanner_a", "scanner_b"):
        output = apply_scanner_profile(
            edge,
            profiles[name],
            pixel_pitch_um=pitch,
            stages=SCANNER_STAGES[:-1],
        )
        edge_rows[name] = [
            _bounded_edge_metrics(
                output[..., channel],
                distance,
                pixel_pitch_um=pitch,
                nyquist_cycles_per_mm=500.0 / pitch,
            )
            for channel in range(3)
        ]

    dark = np.full((32, 32, 3), 1e-12, dtype=np.float64)
    dark_outputs = {
        name: apply_scanner_profile(
            dark,
            profiles[name],
            pixel_pitch_um=pitch,
            stages=("spectral", "dmax"),
        )[0, 0]
        for name in ("scanner_a", "scanner_b")
    }
    floors = {
        name: np.power(10.0, -np.asarray(profiles[name].dmax_density_rgb))
        for name in ("scanner_a", "scanner_b")
    }
    floor_order = np.sign(dark_outputs["scanner_a"] - dark_outputs["scanner_b"])
    expected_order = np.sign(floors["scanner_a"] - floors["scanner_b"])
    dmax_order_matches = np.array_equal(floor_order, expected_order)

    gates = contract["automatic_gates"]
    mtf_values = [
        metric[key]
        for rows in edge_rows.values()
        for metric in rows
        for key in ("mtf50_cycles_per_mm", "mtf10_cycles_per_mm")
        if metric[key] is not None
    ]
    decisions = {
        "identity": identity_exact,
        "repeat": repeat_exact,
        "domain": all(
            np.all(np.isfinite(output))
            and np.all(output >= 0.0)
            and np.all(output <= 1.0)
            for output in outputs.values()
        ),
        "stage_ablation": min(stage_ablation.values())
        >= float(gates["minimum_each_stage_ablation_max_abs"]),
        "scanner_separation": float(np.max(np.abs(scanner_a - scanner_b)))
        >= float(gates["minimum_scanner_a_vs_b_max_abs"]),
        "seed_change": seed_change
        >= float(gates["minimum_seed_change_max_abs"]),
        "noise_boundaries": new_boundary_fraction
        <= float(gates["maximum_noise_new_boundary_fraction"]),
        "nyquist": bool(mtf_values)
        and max(mtf_values) <= 500.0 / pitch + 1e-12,
        "dmax_order": dmax_order_matches,
        "stock_separation": all(
            not hasattr(profile, "stock_id") for profile in profiles.values()
        ),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p6a_scanner_profile_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "identity_exact": identity_exact,
        "stage_ablation_max_abs": stage_ablation,
        "scanner_a_vs_b_max_abs": float(np.max(np.abs(scanner_a - scanner_b))),
        "seed_change_max_abs": seed_change,
        "noise_new_boundary_fraction": new_boundary_fraction,
        "edge_metrics": edge_rows,
        "dark_signal_rgb": {
            name: values.tolist() for name, values in dark_outputs.items()
        },
        "dmax_floor_rgb": {
            name: values.tolist() for name, values in floors.items()
        },
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
