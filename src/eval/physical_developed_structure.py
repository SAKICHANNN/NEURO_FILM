"""Frozen U6.P1B material-structure reference evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.developed_structure import (
    build_bw_silver_context,
    build_colour_dye_cloud_context,
    render_developed_structure,
    render_developed_structure_region,
)


SCHEMA = "neuro_film.u6_p1b_developed_structure_reference_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P1B contract")
    return payload


def evaluate_developed_structure(contract: dict[str, Any]) -> dict[str, Any]:
    height, width = (int(value) for value in contract["input_shape"])
    zoom = int(contract["output_zoom"])
    pitch = float(contract["output_pixel_pitch_um"])
    samples = int(contract["monte_carlo_samples"])
    seed = int(contract["seed"])
    colour_config = contract["colour_dye_cloud"]
    target_cmy = np.asarray(colour_config["target_density_cmy"], dtype=np.float64)
    colour_target = np.broadcast_to(target_cmy, (height, width, 3)).copy()
    colour_context = build_colour_dye_cloud_context(
        colour_target,
        radius_um_cmy=tuple(colour_config["radius_um_cmy"]),
        mark_optical_density_cmy=tuple(
            colour_config["mark_optical_density_cmy"]
        ),
        output_zoom=zoom,
        output_pixel_pitch_um=pitch,
        monte_carlo_samples=samples,
        seed=seed,
    )
    colour = render_developed_structure(colour_context)
    bw_config = contract["bw_metallic_silver"]
    bw_target_density = float(bw_config["target_density"])
    bw_context = build_bw_silver_context(
        np.full((height, width), bw_target_density, dtype=np.float64),
        radius_um=float(bw_config["radius_um"]),
        output_zoom=zoom,
        output_pixel_pitch_um=pitch,
        monte_carlo_samples=samples,
        seed=seed + 1,
    )
    bw = render_developed_structure(bw_context)
    margin = zoom
    colour_interior = colour.values[margin:-margin, margin:-margin]
    bw_interior = bw.values[margin:-margin, margin:-margin, 0]
    target_bw_transmission = 10.0**-bw_target_density
    colour_mean = np.mean(colour_interior, axis=(0, 1), dtype=np.float64)
    bw_mean = float(np.mean(bw_interior, dtype=np.float64))

    split = colour.values.shape[0] // 2
    colour_parts = [
        render_developed_structure_region(
            colour_context,
            output_origin_yx=(0, 0),
            output_shape=(split, colour.values.shape[1]),
        ).values,
        render_developed_structure_region(
            colour_context,
            output_origin_yx=(split, 0),
            output_shape=(colour.values.shape[0] - split, colour.values.shape[1]),
        ).values,
    ]
    colour_partition_exact = np.array_equal(
        np.concatenate(colour_parts, axis=0), colour.values
    )
    repeat_colour = render_developed_structure(colour_context).values
    repeat_bw = render_developed_structure(bw_context).values
    metrics = {
        "colour_context_fingerprint": colour_context.fingerprint(),
        "bw_context_fingerprint": bw_context.fingerprint(),
        "colour_grain_count_cmy": [
            len(layer) for layer in colour_context.centers_by_layer
        ],
        "bw_grain_count": len(bw_context.centers_by_layer[0]),
        "colour_interior_mean_density_cmy": colour_mean.tolist(),
        "colour_interior_mean_density_abs_error_max": float(
            np.max(np.abs(colour_mean - target_cmy))
        ),
        "colour_interior_variance_cmy": np.var(
            colour_interior, axis=(0, 1), dtype=np.float64
        ).tolist(),
        "bw_target_transmittance": target_bw_transmission,
        "bw_interior_mean_transmittance": bw_mean,
        "bw_interior_mean_transmittance_abs_error": abs(
            bw_mean - target_bw_transmission
        ),
        "bw_interior_variance": float(np.var(bw_interior, dtype=np.float64)),
        "colour_partition_exact": colour_partition_exact,
        "repeat_colour_exact": np.array_equal(repeat_colour, colour.values),
        "repeat_bw_exact": np.array_equal(repeat_bw, bw.values),
        "finite_domain_valid": bool(
            np.all(np.isfinite(colour.values))
            and np.all(colour.values >= 0.0)
            and np.all(np.isfinite(bw.values))
            and np.all(bw.values > 0.0)
            and np.all(bw.values <= 1.0)
        ),
    }
    gates = contract["automatic_gates"]
    decisions = {
        "colour_mean": metrics["colour_interior_mean_density_abs_error_max"]
        <= float(gates["colour_interior_mean_density_abs_error_max"]),
        "bw_mean": metrics["bw_interior_mean_transmittance_abs_error"]
        <= float(gates["bw_interior_mean_transmittance_abs_error_max"]),
        "colour_variance": min(metrics["colour_interior_variance_cmy"])
        >= float(gates["minimum_structure_variance"]),
        "bw_variance": metrics["bw_interior_variance"]
        >= float(gates["minimum_structure_variance"]),
        "repeat": metrics["repeat_colour_exact"] and metrics["repeat_bw_exact"],
        "partition": metrics["colour_partition_exact"],
        "domain": metrics["finite_domain_valid"],
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p1b_developed_structure_reference_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
        "negative_control": contract["frozen_negative_control"],
    }
    identity = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": identity}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
