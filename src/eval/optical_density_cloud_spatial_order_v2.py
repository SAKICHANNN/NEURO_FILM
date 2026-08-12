"""Frozen U6.P4DQ optical-density cloud/spatial ordering audit."""

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
        if _sha(parent) != binding["sha256"]:
            raise RuntimeError("P4DQ parent drift")
        if (
            "required_decision" in binding
            and payload.get("decision") != binding["required_decision"]
        ):
            raise RuntimeError("P4DQ parent decision drift")
        if (
            "required_status" in binding
            and payload.get("status") != binding["required_status"]
        ):
            raise RuntimeError("P4DQ parent status drift")
    return contract


def _render(
    profile: CrossLayerCloudReferenceProfile,
    target: np.ndarray,
    *,
    seed: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    results = tuple(
        iter_optical_density_cross_layer_cloud_rows_v2(
            profile, target, seed=seed, row_tile_height=rows
        )
    )
    return (
        np.concatenate([result.density for _, result in results]).astype(np.float64),
        np.concatenate([result.transmittance for _, result in results]).astype(
            np.float64
        ),
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

    def adjacent(values: np.ndarray) -> np.ndarray:
        return apply_bounded_development_adjacency(
            values,
            spatial,
            maximum_absolute_transmittance_delta=p5c[
                "maximum_absolute_transmittance_delta"
            ],
            maximum_absolute_density_delta=p5c["maximum_absolute_density_delta"],
        )

    diffusion_only = apply_dye_diffusion(developed, spatial)
    precloud_target = apply_dye_diffusion(adjacent(developed), spatial)
    profile = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    pre_density, pre_t = _render(
        profile,
        precloud_target,
        seed=fixture["seed"],
        rows=fixture["height"],
    )
    tiled_density, tiled_t = _render(
        profile,
        precloud_target,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    repeat_density, repeat_t = _render(
        profile,
        precloud_target,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    raw_density, _ = _render(
        profile,
        developed,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    post_density = apply_dye_diffusion(adjacent(raw_density), spatial)
    post_t = np.power(10.0, -post_density)
    expected_t = np.power(10.0, -precloud_target)
    margin = fixture["measurement_margin"]
    midpoint = fixture["width"] // 2
    flat_regions = (
        np.s_[margin:-margin, margin : midpoint - 3 * margin, :],
        np.s_[margin:-margin, midpoint + 3 * margin : -margin, :],
    )
    mean_error = max(
        float(
            np.max(
                np.abs(
                    np.mean(pre_t[region], axis=(0, 1), dtype=np.float64)
                    - np.mean(expected_t[region], axis=(0, 1), dtype=np.float64)
                )
            )
        )
        for region in flat_regions
    )
    pre_noise = np.concatenate(
        [(pre_t - expected_t)[region].ravel() for region in flat_regions]
    )
    post_noise = np.concatenate(
        [(post_t - expected_t)[region].ravel() for region in flat_regions]
    )
    pre_over, pre_under = _edge_excursion(pre_t)
    post_over, post_under = _edge_excursion(post_t)
    new_boundary = float(
        np.mean(
            ((pre_t <= 0.0) | (pre_t >= 1.0))
            & ~((expected_t <= 0.0) | (expected_t >= 1.0))
        )
    )
    p4do = json.loads(
        (
            root / "docs/evidence/U6_P4DO_CLOUD_SPATIAL_ORDER_ABLATION_RESULT.json"
        ).read_text()
    )
    metrics = {
        "precloud_mean_transmittance_error": mean_error,
        "precloud_noise_rms": _rms(pre_noise),
        "postcloud_noise_rms": _rms(post_noise),
        "postcloud_to_precloud_noise_ratio": _rms(post_noise) / _rms(pre_noise),
        "precloud_edge_overshoot": pre_over,
        "precloud_edge_undershoot": pre_under,
        "postcloud_edge_overshoot": post_over,
        "postcloud_edge_undershoot": post_under,
        "postcloud_edge_excursion_increase": max(
            post_over - pre_over, post_under - pre_under, 0.0
        ),
        "precloud_new_boundary_fraction": new_boundary,
        "adjacency_stage_rms_before_cloud": _rms(
            np.power(10.0, -precloud_target) - np.power(10.0, -diffusion_only)
        ),
    }
    gates = contract["gates"]
    decisions = {
        "mean": mean_error <= gates["maximum_precloud_mean_transmittance_error"],
        "noise": metrics["postcloud_to_precloud_noise_ratio"]
        <= gates["maximum_postcloud_to_precloud_noise_ratio"],
        "pre_overshoot": pre_over <= gates["maximum_precloud_edge_overshoot"],
        "pre_undershoot": pre_under <= gates["maximum_precloud_edge_undershoot"],
        "post_excursion": metrics["postcloud_edge_excursion_increase"]
        <= gates["maximum_postcloud_edge_excursion_increase"],
        "boundary": new_boundary <= gates["maximum_new_boundary_fraction"],
        "adjacency_nonzero": metrics["adjacency_stage_rms_before_cloud"]
        >= gates["minimum_adjacency_stage_rms_before_cloud"],
        "partition": bool(
            np.array_equal(pre_density, tiled_density)
            and np.array_equal(pre_t, tiled_t)
        ),
        "repeat": bool(
            np.array_equal(tiled_density, repeat_density)
            and np.array_equal(tiled_t, repeat_t)
        ),
        "p4do_closed": p4do["automatic_pass"] is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "metrics": metrics,
        "precloud_transmittance_sha256": hashlib.sha256(
            pre_t.astype("<f8").tobytes()
        ).hexdigest(),
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dq_optical_density_cloud_spatial_order_report.v2",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
