"""Frozen U6.P4DS target-resolution compound-Poisson cloud LOD."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cloud_spatial_order_ablation import _edge_excursion
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_aperture_lod import block_aperture_mean
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_optical_density_cross_layer_cloud_rows_v2,
    optical_density_capacity_cmy,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
        or evidence["selected_factor"] != contract["parent"]["required_factor"]
    ):
        raise RuntimeError("P4DS parent drift")
    return contract


def compile_target_resolution_profile(
    profile: CrossLayerCloudReferenceProfile,
    factor: int,
    *,
    rate_multiplier: float | None = None,
) -> CrossLayerCloudReferenceProfile:
    """Compile area-summed rates and area-averaged marks at target resolution."""

    area = factor * factor if rate_multiplier is None else rate_multiplier
    if not np.isfinite(area) or area < 1:
        raise ValueError("rate_multiplier must be finite and at least one")
    count = profile.count_profile
    compiled_count = replace(
        count,
        marginal_rates_cmy=tuple(value * area for value in count.marginal_rates_cmy),
        shared_all_rate=count.shared_all_rate * area,
        shared_pair_rates_cm_cy_my=tuple(
            value * area for value in count.shared_pair_rates_cm_cy_my
        ),
        mark_optical_density_cmy=tuple(
            value / area for value in count.mark_optical_density_cmy
        ),
    )
    return replace(
        profile,
        count_profile=compiled_count,
        gaussian_sigma_pixels_cmy=tuple(
            value / factor for value in profile.gaussian_sigma_pixels_cmy
        ),
    )


def _render(
    profile: CrossLayerCloudReferenceProfile,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    return np.concatenate(
        [
            result.transmittance
            for _, result in iter_optical_density_cross_layer_cloud_rows_v2(
                profile,
                target,
                seed=seed,
                row_tile_height=127,
            )
        ]
    ).astype(np.float64)


def _noise_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, 3)
    rms = np.sqrt(np.mean(np.square(flat), axis=0, dtype=np.float64))
    correlation = np.corrcoef(flat, rowvar=False)
    return rms, correlation


def _radial_nps(values: np.ndarray, bins: int = 16) -> np.ndarray:
    rows = []
    height, width = values.shape[:2]
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    edges = np.linspace(0.0, float(np.max(radius)), bins + 1)
    for channel in range(3):
        centered = values[..., channel] - np.mean(values[..., channel])
        power = np.square(np.abs(np.fft.rfft2(centered)))
        curve = []
        for index in range(bins):
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
            curve.append(float(np.mean(power[mask])) if np.any(mask) else 0.0)
        curve_array = np.asarray(curve, dtype=np.float64)
        total = float(np.sum(curve_array))
        rows.append(curve_array / total if total > 0.0 else curve_array)
    return np.stack(rows)


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    factor = fixture["aperture_factor"]
    shape = (fixture["output_height"], fixture["output_width"])
    high_shape = (shape[0] * factor, shape[1] * factor)
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    candidate = compile_target_resolution_profile(reference, factor)
    common_capacity = np.minimum(
        np.asarray(optical_density_capacity_cmy(reference)),
        np.asarray(optical_density_capacity_cmy(candidate)),
    )
    target_value = common_capacity * fixture["flat_density_fraction"]
    target = np.broadcast_to(target_value, (*shape, 3)).copy()
    high_target = np.broadcast_to(target_value, (*high_shape, 3)).copy()
    expected = np.power(10.0, -target_value)
    confirmation = []
    for seed in fixture["confirmation_seeds"]:
        high = block_aperture_mean(_render(reference, high_target, seed), factor)
        low = _render(candidate, target, seed)
        high_noise = high - expected
        low_noise = low - expected
        high_rms, high_corr = _noise_stats(high_noise)
        low_rms, low_corr = _noise_stats(low_noise)
        index = np.triu_indices(3, 1)
        confirmation.append(
            {
                "seed": seed,
                "mean_transmittance_absolute_error": float(
                    np.max(
                        np.abs(np.mean(low, axis=(0, 1)) - np.mean(high, axis=(0, 1)))
                    )
                ),
                "marginal_noise_rms_relative_error": float(
                    np.max(np.abs(low_rms - high_rms) / high_rms)
                ),
                "cross_channel_correlation_absolute_error": float(
                    np.max(np.abs(low_corr[index] - high_corr[index]))
                ),
                "radial_nps_shape_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(_radial_nps(low_noise) - _radial_nps(high_noise))
                        )
                    )
                ),
                "low_sha256": hashlib.sha256(low.astype("<f8").tobytes()).hexdigest(),
            }
        )
    edge_target = np.empty((*shape, 3), dtype=np.float64)
    edge_target[:, : shape[1] // 2] = common_capacity * 0.2
    edge_target[:, shape[1] // 2 :] = common_capacity * 0.8
    high_edge_target = np.repeat(np.repeat(edge_target, factor, axis=0), factor, axis=1)
    reference_edge = block_aperture_mean(
        _render(reference, high_edge_target, fixture["seed"]), factor
    )
    candidate_edge = _render(candidate, edge_target, fixture["seed"])
    ref_over, ref_under = _edge_excursion(reference_edge)
    cand_over, cand_under = _edge_excursion(candidate_edge)
    candidate_repeat = _render(candidate, edge_target, fixture["seed"])
    gates = contract["gates"]
    worst = {
        "mean_transmittance_absolute_error": max(
            row["mean_transmittance_absolute_error"] for row in confirmation
        ),
        "marginal_noise_rms_relative_error": max(
            row["marginal_noise_rms_relative_error"] for row in confirmation
        ),
        "cross_channel_correlation_absolute_error": max(
            row["cross_channel_correlation_absolute_error"] for row in confirmation
        ),
        "radial_nps_shape_rmse": max(
            row["radial_nps_shape_rmse"] for row in confirmation
        ),
        "edge_overshoot_absolute_difference": abs(cand_over - ref_over),
        "edge_undershoot_absolute_difference": abs(cand_under - ref_under),
        "new_boundary_fraction": float(
            np.mean(
                ((candidate_edge <= 0.0) | (candidate_edge >= 1.0))
                & ~((reference_edge <= 0.0) | (reference_edge >= 1.0))
            )
        ),
    }
    decisions = {
        "mean": worst["mean_transmittance_absolute_error"]
        <= gates["maximum_mean_transmittance_absolute_error"],
        "rms": worst["marginal_noise_rms_relative_error"]
        <= gates["maximum_marginal_noise_rms_relative_error"],
        "correlation": worst["cross_channel_correlation_absolute_error"]
        <= gates["maximum_cross_channel_correlation_absolute_error"],
        "nps": worst["radial_nps_shape_rmse"] <= gates["maximum_radial_nps_shape_rmse"],
        "overshoot": worst["edge_overshoot_absolute_difference"]
        <= gates["maximum_edge_overshoot_absolute_difference"],
        "undershoot": worst["edge_undershoot_absolute_difference"]
        <= gates["maximum_edge_undershoot_absolute_difference"],
        "boundary": worst["new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "repeat": bool(np.array_equal(candidate_edge, candidate_repeat)),
        "product_disabled": candidate.product_enabled is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "reference_profile_identity": reference.identity(),
        "candidate_profile_identity": candidate.identity(),
        "candidate_optical_density_capacity_cmy": list(
            optical_density_capacity_cmy(candidate)
        ),
        "confirmation_worst": worst,
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4ds_target_resolution_cloud_lod_report.v2",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "confirmation": confirmation,
    }


__all__ = ["compile_target_resolution_profile", "evaluate", "load_contract"]
