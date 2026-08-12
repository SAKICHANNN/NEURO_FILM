"""Frozen U6.P4DY compiled cloud profile restart audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.compiled_cloud_attenuation_row_runtime import _render
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    CompiledCloudAttenuationProfile,
    compile_cloud_attenuation_profile,
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
        raise RuntimeError("P4DY parent drift")
    return contract


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    reference = CrossLayerCloudReferenceProfile.from_payload(
        evaluate_capacity(
            root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
        )["compiled_profile"]
    )
    parent = json.loads(
        (root / contract["parent"]["path"]).read_text(encoding="utf-8")
    )
    p4dw = json.loads(
        (
            root
            / "docs/evidence/U6_P4DW_NPS_PRESERVING_CLOUD_RESIDUAL_ATTENUATION_V6_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    profile = compile_cloud_attenuation_profile(
        reference,
        aperture_factor=4,
        base_rate_multiplier=16.0,
        channel_residual_gain=tuple(p4dw["compiled_channel_gain"]),
    )
    payload = profile.to_payload()
    restarted = CompiledCloudAttenuationProfile.from_payload(
        json.loads(_canonical(payload))
    )
    shape = (fixture["height"], fixture["width"])
    capacity = np.asarray(optical_density_capacity_cmy(profile.base_profile))
    yy, xx = np.indices(shape, dtype=np.float64)
    target = (0.25 + 0.5 * (xx + yy) / (sum(shape) - 2))[..., None] * capacity
    original_density, original_t = _render(
        profile, target, fixture["seed"], fixture["row_partition_height"]
    )
    restart_density, restart_t = _render(
        restarted, target, fixture["seed"], fixture["row_partition_height"]
    )

    unknown = copy.deepcopy(payload)
    unknown["unknown"] = True
    product = copy.deepcopy(payload)
    product["product_enabled"] = True
    base_drift = copy.deepcopy(payload)
    base_drift["base_profile"]["marginal_count_rates_cmy"][0] += 0.01
    rejections = []
    for candidate in (unknown, product, base_drift):
        try:
            CompiledCloudAttenuationProfile.from_payload(candidate)
        except ValueError:
            rejections.append(True)
        else:
            rejections.append(False)
    gain_drift = copy.deepcopy(payload)
    gain_drift["channel_residual_gain"][0] -= 0.01
    drift_profile = CompiledCloudAttenuationProfile.from_payload(gain_drift)
    decisions = {
        "payload": bool(
            restarted.identity() == profile.identity()
            and restarted.to_payload() == payload
        ),
        "pixels": bool(
            np.array_equal(original_density, restart_density)
            and np.array_equal(original_t, restart_t)
        ),
        "unknown": rejections[0],
        "base_drift": rejections[2],
        "gain_drift": drift_profile.identity() != profile.identity(),
        "product": rejections[1],
        "parent": parent["automatic_pass"] is True,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "profile_identity": profile.identity(),
        "canonical_payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "restart_transmittance_sha256": hashlib.sha256(
            restart_t.astype("<f4").tobytes()
        ).hexdigest(),
        "gates": decisions,
        "decision": contract["decision_if_pass"] if all(decisions.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dy_compiled_cloud_profile_restart_report.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
