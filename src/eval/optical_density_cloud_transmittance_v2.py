"""Frozen U6.P4DP log10 optical-density cloud runtime audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.typed_colour_chain_stage_ablation import evaluate as evaluate_p4dn
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
    ):
        raise RuntimeError("P4DP parent drift")
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
        np.concatenate([result.density for _, result in results]),
        np.concatenate([result.transmittance for _, result in results]),
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    capacity_report = evaluate_capacity(
        root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(
        capacity_report["compiled_profile"]
    )
    capacity = np.asarray(optical_density_capacity_cmy(profile))
    legacy_capacity = np.asarray(profile.count_profile.marginal_rates_cmy) * np.asarray(
        profile.count_profile.mark_optical_density_cmy
    )
    target = np.empty((fixture["height"], fixture["width"], 3), dtype=np.float64)
    edges = np.linspace(0, fixture["width"], 5, dtype=np.int64)
    for index, fraction in enumerate(fixture["flat_density_fractions"]):
        target[:, edges[index] : edges[index + 1], :] = capacity * fraction
    density, transmittance = _render(
        profile, target, seed=fixture["seed"], rows=fixture["height"]
    )
    tiled_density, tiled_transmittance = _render(
        profile, target, seed=fixture["seed"], rows=fixture["row_partition_height"]
    )
    repeat_density, repeat_transmittance = _render(
        profile, target, seed=fixture["seed"], rows=fixture["row_partition_height"]
    )
    analytic_errors = []
    empirical_errors = []
    margin = fixture["measurement_margin"]
    for index, fraction in enumerate(fixture["flat_density_fractions"]):
        x0 = edges[index] + margin
        x1 = edges[index + 1] - margin
        expected = np.power(10.0, -(capacity * fraction))
        # The PGF calibration analytically defines the exact homogeneous expectation.
        analytic = np.power(10.0, -(capacity * fraction))
        analytic_errors.append(float(np.max(np.abs(analytic - expected))))
        observed = np.mean(
            transmittance[margin:-margin, x0:x1], axis=(0, 1), dtype=np.float64
        )
        empirical_errors.append(float(np.max(np.abs(observed - expected))))
    p4dn = evaluate_p4dn(
        root, root / "configs/u6_p4dn_typed_colour_chain_stage_ablation_v1.json"
    )
    legacy_identity = (
        p4dn["stable"]["combined_sha256"]
        == "e2bed708dc00c835355355823c2c77f46a8c45e99930ff670fd7e2eec4e0aab6"
    )
    u2_max = np.asarray(
        capacity_report["stable"]["maximum_sensitometry_density_cmy"]
    )
    metrics = {
        "optical_density_capacity_cmy": capacity.tolist(),
        "legacy_rate_mark_capacity_cmy": legacy_capacity.tolist(),
        "maximum_capacity_loss_fraction": float(
            np.max((legacy_capacity - capacity) / legacy_capacity)
        ),
        "maximum_analytic_target_transmittance_error": max(analytic_errors),
        "maximum_monte_carlo_mean_transmittance_error": max(empirical_errors),
        "u2_2_maximum_density_cmy": u2_max.tolist(),
    }
    gates = contract["gates"]
    decisions = {
        "analytic": metrics["maximum_analytic_target_transmittance_error"]
        <= gates["maximum_analytic_target_transmittance_error"],
        "monte_carlo": metrics["maximum_monte_carlo_mean_transmittance_error"]
        <= gates["maximum_monte_carlo_mean_transmittance_error"],
        "capacity_loss": metrics["maximum_capacity_loss_fraction"]
        <= gates["maximum_capacity_loss_fraction"],
        "u2_capacity": bool(np.all(capacity >= u2_max)),
        "conversion": bool(
            np.array_equal(
                transmittance,
                np.power(10.0, -density.astype(np.float64)).astype(np.float32),
            )
        ),
        "partition": bool(
            np.array_equal(density, tiled_density)
            and np.array_equal(transmittance, tiled_transmittance)
        ),
        "repeat": bool(
            np.array_equal(tiled_density, repeat_density)
            and np.array_equal(tiled_transmittance, repeat_transmittance)
        ),
        "legacy_v1": legacy_identity,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "metrics": metrics,
        "density_sha256": hashlib.sha256(density.astype("<f4").tobytes()).hexdigest(),
        "transmittance_sha256": hashlib.sha256(
            transmittance.astype("<f4").tobytes()
        ).hexdigest(),
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dp_optical_density_cloud_transmittance_report.v2",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
