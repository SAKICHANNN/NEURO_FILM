"""Frozen U6.P4DX compiled attenuation row-runtime audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    compile_cloud_attenuation_profile,
    estimate_compiled_cloud_attenuation_live_bytes,
    iter_compiled_cloud_attenuation_rows,
)
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import optical_density_capacity_cmy


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
        raise RuntimeError("P4DX parent drift")
    return contract


def _render(profile, target: np.ndarray, seed: int, rows: int):
    results = tuple(
        iter_compiled_cloud_attenuation_rows(
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
    compiled = contract["compiled_profile"]
    shape = (fixture["height"], fixture["width"])
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    profile = compile_cloud_attenuation_profile(
        reference,
        aperture_factor=compiled["aperture_factor"],
        base_rate_multiplier=compiled["base_rate_multiplier"],
        channel_residual_gain=tuple(compiled["channel_residual_gain"]),
    )
    capacity = np.asarray(optical_density_capacity_cmy(profile.base_profile))
    yy, xx = np.indices(shape, dtype=np.float64)
    modulation = 0.35 + 0.25 * (xx / max(shape[1] - 1, 1)) + 0.2 * (
        yy / max(shape[0] - 1, 1)
    )
    target = modulation[..., None] * capacity
    seed = fixture["seed"]
    full_density, full_t = _render(
        profile, target, seed, fixture["full_partition_height"]
    )
    rows = []
    maximum_live_bytes = 0
    all_exact = True
    for partition in fixture["candidate_partition_heights"]:
        density, transmittance = _render(profile, target, seed, partition)
        exact = bool(
            np.array_equal(density, full_density)
            and np.array_equal(transmittance, full_t)
        )
        live_bytes = estimate_compiled_cloud_attenuation_live_bytes(
            profile, full_shape=shape, row_tile_height=partition
        )
        maximum_live_bytes = max(maximum_live_bytes, live_bytes)
        all_exact &= exact
        rows.append(
            {
                "partition_height": partition,
                "exact": exact,
                "estimated_live_bytes": live_bytes,
            }
        )
    repeat_density, repeat_t = _render(profile, target, seed, 31)
    invalid_rejections = []
    invalids = [
        (target, -1, 31),
        (target, seed, 0),
        (np.full_like(target, np.nan), seed, 31),
        (target * 100.0, seed, 31),
    ]
    for invalid_target, invalid_seed, invalid_rows in invalids:
        iterator = iter_compiled_cloud_attenuation_rows(
            profile,
            invalid_target,
            seed=invalid_seed,
            row_tile_height=invalid_rows,
        )
        try:
            next(iterator)
        except ValueError:
            invalid_rejections.append(True)
        else:
            invalid_rejections.append(False)
    roundtrip = np.power(10.0, -full_density.astype(np.float64)).astype(np.float32)
    gates = contract["gates"]
    decisions = {
        "partition": all_exact,
        "repeat": bool(
            np.array_equal(repeat_density, full_density)
            and np.array_equal(repeat_t, full_t)
        ),
        "domain": bool(np.all((full_t > 0.0) & (full_t < 1.0))),
        "roundtrip": bool(np.array_equal(roundtrip, full_t)),
        "workspace": maximum_live_bytes <= gates["maximum_estimated_live_bytes"],
        "invalid": all(invalid_rejections),
        "product_disabled": profile.product_enabled is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "density_sha256": hashlib.sha256(full_density.astype("<f4").tobytes()).hexdigest(),
        "transmittance_sha256": hashlib.sha256(full_t.astype("<f4").tobytes()).hexdigest(),
        "maximum_estimated_live_bytes": maximum_live_bytes,
        "partition_results": rows,
        "gates": decisions,
        "decision": contract["decision_if_pass"] if all(decisions.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dx_compiled_cloud_attenuation_row_runtime_report.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
