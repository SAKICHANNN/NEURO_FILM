"""Frozen U6.P4DO cloud/spatial ordering ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_spatial_response import _profile, _slanted_edge
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_target_density_cross_layer_cloud_rows,
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
            raise RuntimeError("P4DO parent drift")
        if "required_decision" in binding and (
            payload.get("decision") != binding["required_decision"]
        ):
            raise RuntimeError("P4DO parent decision drift")
        if "required_status" in binding and (
            payload.get("status") != binding["required_status"]
        ):
            raise RuntimeError("P4DO parent status drift")
    return contract


def _sample(
    profile: CrossLayerCloudReferenceProfile,
    density: np.ndarray,
    maximum: np.ndarray,
    *,
    seed: int,
    rows: int,
) -> np.ndarray:
    return np.concatenate(
        [
            result.density
            for _, result in iter_target_density_cross_layer_cloud_rows(
                profile,
                density,
                maximum_developed_density_cmy=tuple(maximum),
                seed=seed,
                row_tile_height=rows,
            )
        ]
    ).astype(np.float64)


def _edge_excursion(values: np.ndarray) -> tuple[float, float]:
    esf = np.mean(values, axis=0)
    edge = esf.shape[0] // 2
    left = np.mean(esf[8 : max(9, edge - 24)], axis=0)
    right = np.mean(esf[min(edge + 24, esf.shape[0] - 9) : -8], axis=0)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    scale = np.maximum(high - low, 1e-12)
    overshoot = float(np.max(np.maximum((esf - high) / scale, 0.0)))
    undershoot = float(np.max(np.maximum((low - esf) / scale, 0.0)))
    return overshoot, undershoot


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    exposure, _ = _slanted_edge(shape, 5.0, 0.08, 2.5)
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    developed = operator.apply(exposure)
    spatial_contract = json.loads(
        (root / "configs/u6_p5a_spatial_response_primitives_v1.json").read_text()
    )
    spatial = _profile(spatial_contract)
    p5c = json.loads((root / "configs/u6_p5c_bounded_adjacency_v1.json").read_text())[
        "candidate"
    ]
    bounded = apply_bounded_development_adjacency(
        developed,
        spatial,
        maximum_absolute_transmittance_delta=p5c[
            "maximum_absolute_transmittance_delta"
        ],
        maximum_absolute_density_delta=p5c["maximum_absolute_density_delta"],
    )
    precloud_mean = apply_dye_diffusion(bounded, spatial)
    diffusion_only_mean = apply_dye_diffusion(developed, spatial)
    capacity = evaluate_capacity(
        root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    maximum = np.asarray(profile.count_profile.marginal_rates_cmy) * np.asarray(
        profile.count_profile.mark_optical_density_cmy
    )
    precloud_density = _sample(
        profile,
        precloud_mean,
        maximum,
        seed=fixture["seed"],
        rows=fixture["height"],
    )
    precloud_tiled = _sample(
        profile,
        precloud_mean,
        maximum,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    precloud_repeat = _sample(
        profile,
        precloud_mean,
        maximum,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    raw_cloud = _sample(
        profile,
        developed,
        maximum,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    postcloud_density = apply_dye_diffusion(
        apply_bounded_development_adjacency(
            raw_cloud,
            spatial,
            maximum_absolute_transmittance_delta=p5c[
                "maximum_absolute_transmittance_delta"
            ],
            maximum_absolute_density_delta=p5c["maximum_absolute_density_delta"],
        ),
        spatial,
    )
    diffusion_cloud = _sample(
        profile,
        diffusion_only_mean,
        maximum,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    pre_t = np.exp(-precloud_density)
    pre_mean_t = np.exp(-precloud_mean)
    post_t = np.exp(-postcloud_density)
    post_mean_t = np.exp(-precloud_mean)
    raw_t = np.exp(-raw_cloud)
    raw_mean_t = np.exp(-developed)
    diffusion_t = np.exp(-diffusion_cloud)
    diffusion_mean_t = np.exp(-diffusion_only_mean)
    margin = fixture["measurement_margin"]
    region = np.s_[margin:-margin, margin:-margin, :]
    pre_noise = _rms((pre_t - pre_mean_t)[region])
    post_noise = _rms((post_t - post_mean_t)[region])
    raw_noise = _rms((raw_t - raw_mean_t)[region])
    mean_error = float(
        max(
            np.max(np.abs(np.mean(pre_t, axis=0) - np.mean(pre_mean_t, axis=0))),
            np.max(np.abs(np.mean(post_t, axis=0) - np.mean(post_mean_t, axis=0))),
            np.max(
                np.abs(np.mean(diffusion_t, axis=0) - np.mean(diffusion_mean_t, axis=0))
            ),
        )
    )
    overshoot, undershoot = _edge_excursion(pre_t)
    new_boundary = float(
        np.mean(
            ((pre_t <= 0.0) | (pre_t >= 1.0))
            & ~((pre_mean_t <= 0.0) | (pre_mean_t >= 1.0))
        )
    )
    metrics = {
        "mean_transmittance_error": mean_error,
        "raw_cloud_noise_rms": raw_noise,
        "precloud_noise_rms": pre_noise,
        "postcloud_noise_rms": post_noise,
        "postcloud_to_precloud_noise_ratio": post_noise / pre_noise,
        "postcloud_to_raw_noise_ratio": post_noise / raw_noise,
        "precloud_edge_overshoot": overshoot,
        "precloud_edge_undershoot": undershoot,
        "precloud_new_boundary_fraction": new_boundary,
        "adjacency_stage_rms_before_cloud": _rms(
            (np.exp(-precloud_mean) - np.exp(-diffusion_only_mean))[region]
        ),
    }
    gates = contract["gates"]
    decisions = {
        "mean": mean_error <= gates["maximum_mean_transmittance_error"],
        "noise_amplification": post_noise / pre_noise
        <= gates["maximum_noise_rms_amplification_after_cloud"],
        "overshoot": overshoot <= gates["maximum_edge_overshoot"],
        "undershoot": undershoot <= gates["maximum_edge_undershoot"],
        "boundary": new_boundary <= gates["maximum_new_boundary_fraction"],
        "adjacency_nonzero": metrics["adjacency_stage_rms_before_cloud"]
        >= gates["minimum_adjacency_stage_rms_before_cloud"],
        "partition": bool(np.array_equal(precloud_density, precloud_tiled)),
        "repeat": bool(np.array_equal(precloud_tiled, precloud_repeat)),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "metrics": metrics,
        "precloud_density_sha256": hashlib.sha256(
            precloud_density.astype("<f8").tobytes()
        ).hexdigest(),
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4do_cloud_spatial_order_ablation_report.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
