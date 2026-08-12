"""Frozen U6.P4DR optical-density cloud aperture-LOD compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cloud_spatial_order_ablation import _edge_excursion
from src.eval.physical_spatial_response import _profile, _slanted_edge
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.cross_layer_aperture_lod import block_aperture_mean
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_optical_density_cross_layer_cloud_rows_v2,
)
from src.film_physics.spatial_response import (
    apply_bounded_development_adjacency,
    apply_dye_diffusion,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    for binding in contract["parents"]:
        parent = root / binding["path"]
        payload = json.loads(parent.read_text(encoding="utf-8"))
        if (
            _sha(parent) != binding["sha256"]
            or payload.get("decision") != binding["required_decision"]
        ):
            raise RuntimeError("P4DR parent drift")
    return contract


def _flat_mean_error(
    observed: np.ndarray, expected: np.ndarray, *, margin: int
) -> float:
    midpoint = observed.shape[1] // 2
    regions = (
        np.s_[margin:-margin, margin : midpoint - 3 * margin, :],
        np.s_[margin:-margin, midpoint + 3 * margin : -margin, :],
    )
    return max(
        float(
            np.max(
                np.abs(
                    np.mean(observed[region], axis=(0, 1), dtype=np.float64)
                    - np.mean(expected[region], axis=(0, 1), dtype=np.float64)
                )
            )
        )
        for region in regions
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    exposure, _ = _slanted_edge(
        shape,
        fixture["slant_degrees"],
        fixture["low_high_exposure"][0],
        fixture["low_high_exposure"][1],
    )
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    developed = operator.apply(exposure)
    spatial = _profile(
        json.loads(
            (root / "configs/u6_p5a_spatial_response_primitives_v1.json").read_text()
        )
    )
    p5c = json.loads((root / "configs/u6_p5c_bounded_adjacency_v1.json").read_text())[
        "candidate"
    ]
    adjacent = apply_bounded_development_adjacency(
        developed,
        spatial,
        maximum_absolute_transmittance_delta=p5c[
            "maximum_absolute_transmittance_delta"
        ],
        maximum_absolute_density_delta=p5c["maximum_absolute_density_delta"],
    )
    target = apply_dye_diffusion(adjacent, spatial)
    expected = np.power(10.0, -target)
    profile = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )

    def render() -> np.ndarray:
        return np.concatenate(
            [
                result.transmittance
                for _, result in iter_optical_density_cross_layer_cloud_rows_v2(
                    profile,
                    target,
                    seed=fixture["seed"],
                    row_tile_height=127,
                )
            ]
        ).astype(np.float64)

    point = render()
    point_repeat = render()
    point_noise = _rms(point - expected)
    rows = []
    gates = contract["gates"]
    for factor in fixture["candidate_aperture_factors"]:
        observed = block_aperture_mean(point, factor)
        repeated = block_aperture_mean(point_repeat, factor)
        reference = block_aperture_mean(expected, factor)
        margin = max(2, fixture["measurement_margin"] // factor)
        overshoot, undershoot = _edge_excursion(observed)
        noise_ratio = _rms(observed - reference) / point_noise
        boundary = float(
            np.mean(
                ((observed <= 0.0) | (observed >= 1.0))
                & ~((reference <= 0.0) | (reference >= 1.0))
            )
        )
        metrics = {
            "factor": factor,
            "mean_transmittance_error": _flat_mean_error(
                observed, reference, margin=margin
            ),
            "edge_overshoot": overshoot,
            "edge_undershoot": undershoot,
            "new_boundary_fraction": boundary,
            "noise_rms_retention_ratio": noise_ratio,
            "repeat_exact": bool(np.array_equal(observed, repeated)),
        }
        decisions = {
            "mean": metrics["mean_transmittance_error"]
            <= gates["maximum_mean_transmittance_error"],
            "overshoot": overshoot <= gates["maximum_edge_overshoot"],
            "undershoot": undershoot <= gates["maximum_edge_undershoot"],
            "boundary": boundary <= gates["maximum_new_boundary_fraction"],
            "noise_min": noise_ratio >= gates["minimum_noise_rms_retention_ratio"],
            "noise_max": noise_ratio <= gates["maximum_noise_rms_retention_ratio"],
            "repeat": metrics["repeat_exact"],
        }
        rows.append(
            {**metrics, "gates": decisions, "automatic_pass": all(decisions.values())}
        )
    selected = next((row for row in rows if row["automatic_pass"]), None)
    p4dq = json.loads(
        (
            root
            / "docs/evidence/U6_P4DQ_OPTICAL_DENSITY_CLOUD_SPATIAL_ORDER_V2_RESULT.json"
        ).read_text()
    )
    all_gates = {
        "candidate": selected is not None,
        "first_pass": selected is None
        or all(not row["automatic_pass"] for row in rows[: rows.index(selected)]),
        "p4dq_closed": p4dq["automatic_pass"] is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "candidates": rows,
        "selected_factor": None if selected is None else selected["factor"],
        "gates": all_gates,
        "decision": contract["decision_if_pass"]
        if all(all_gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dr_optical_density_cloud_aperture_lod_report.v2",
        "automatic_pass": all(all_gates.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
