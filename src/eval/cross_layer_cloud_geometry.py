"""Evaluate exact shared-event continuous dye-cloud geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    build_cross_layer_cloud_geometry,
)
from src.film_physics.developed_structure import (
    render_developed_structure,
    render_developed_structure_region,
)

SCHEMA = "neuro_film.u6_p4cy_cross_layer_cloud_geometry_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cy_cross_layer_cloud_geometry_report.v1"


class CrossLayerCloudGeometryError(RuntimeError):
    pass


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload["parent"]
    pp = root / parent["path"]
    pv = json.loads(pp.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise CrossLayerCloudGeometryError("P4CY contract identity drift")
    if (
        sha256_file(pp) != parent["sha256"]
        or pv.get("decision") != parent["required_decision"]
    ):
        raise CrossLayerCloudGeometryError("P4CY parent drift")
    return payload


def _profile(g: dict[str, Any], seed: int) -> CrossLayerPoissonProfile:
    return CrossLayerPoissonProfile(
        tuple(g["marginal_count_rates_cmy"]),
        g["shared_all_rate"],
        tuple(g["shared_pair_rates_cm_cy_my"]),
        tuple(g["mark_optical_density_cmy"]),
        seed,
        g["component_seed_stride"],
    )


def _row(seed: int, contract: dict[str, Any]) -> dict[str, Any]:
    g = contract["geometry"]
    shape = tuple(g["input_shape"])
    kwargs = {
        "radius_um_cmy": tuple(g["radius_um_cmy"]),
        "output_zoom": g["output_zoom"],
        "output_pixel_pitch_um": g["output_pixel_pitch_um"],
        "monte_carlo_samples": g["monte_carlo_samples"],
    }
    profile = _profile(g, seed)
    geometry = build_cross_layer_cloud_geometry(profile, shape, **kwargs)
    repeat = build_cross_layer_cloud_geometry(profile, shape, **kwargs)
    component_counts = np.array([len(v) for v in geometry.component_centers])
    area = np.prod(shape)
    rates = component_counts / area
    expected = np.array(
        [
            g["shared_all_rate"],
            *g["shared_pair_rates_cm_cy_my"],
            *profile.independent_rates_cmy,
        ]
    )
    full = render_developed_structure(geometry.context).values
    parts = []
    rows = g["render_row_partition"]
    for y in range(0, full.shape[0], rows):
        parts.append(
            render_developed_structure_region(
                geometry.context,
                output_origin_yx=(y, 0),
                output_shape=(min(rows, full.shape[0] - y), full.shape[1]),
            ).values
        )
    return {
        "seed": seed,
        "context_fingerprint": geometry.context.fingerprint(),
        "repeat_context_fingerprint": repeat.context.fingerprint(),
        "maximum_component_rate_relative_error": float(
            np.max(np.abs(rates - expected) / expected)
        ),
        "full_tiled_exact": bool(np.array_equal(full, np.concatenate(parts))),
        "density_minimum": float(np.min(full)),
        "density_maximum": float(np.max(full)),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    c = load_contract(root, contract_path)
    rows = [_row(int(seed), c) for seed in c["geometry"]["seeds"]]
    m = c["metrics"]
    gates = {
        "rates": max(r["maximum_component_rate_relative_error"] for r in rows)
        <= m["maximum_shared_rate_relative_error"],
        "repeat": all(
            r["context_fingerprint"] == r["repeat_context_fingerprint"] for r in rows
        ),
        "tiled": all(r["full_tiled_exact"] for r in rows),
        "nonnegative": all(r["density_minimum"] >= 0 for r in rows),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "maximum_component_rate_relative_error": max(
            r["maximum_component_rate_relative_error"] for r in rows
        ),
        "gates": gates,
        "decision": c["decision_if_pass"] if passed else c["decision_if_fail"],
        "claim_ceiling": c["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "rows": rows,
    }


__all__ = ["CrossLayerCloudGeometryError", "evaluate", "load_contract"]
