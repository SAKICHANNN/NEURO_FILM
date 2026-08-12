"""Frozen U6.P4DH compiler from generic sensitometry to cloud capacity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_cloud_row_stream import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_compound_poisson import CrossLayerPoissonProfile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    chart = contract["parents"]["chart"]
    chart_path = root / chart["path"]
    chart_evidence = json.loads(chart_path.read_text(encoding="utf-8"))
    sensitometry = contract["parents"]["sensitometry"]
    sensitometry_path = root / sensitometry["path"]
    sensitometry_payload = json.loads(sensitometry_path.read_text(encoding="utf-8"))
    if (
        _sha(chart_path) != chart["sha256"]
        or chart_evidence["decision"] != chart["required_decision"]
        or _sha(sensitometry_path) != sensitometry["sha256"]
        or sensitometry_payload["experiment_id"]
        != sensitometry["required_experiment_id"]
    ):
        raise RuntimeError("P4DH parent drift")
    return contract


def compile_profile(
    root: Path, contract: dict[str, Any]
) -> tuple[CrossLayerCloudReferenceProfile, dict[str, Any]]:
    base = _profile(root)
    operator = build_operator(
        json.loads((root / contract["parents"]["sensitometry"]["path"]).read_text())
    )
    compiler = contract["compiler"]
    exposure = np.linspace(
        compiler["linear_exposure_minimum"],
        compiler["linear_exposure_maximum"],
        compiler["samples"],
        dtype=np.float64,
    )
    density = operator.apply(np.repeat(exposure[:, None], 3, axis=1))
    maximum_density = np.max(density, axis=0)
    base_capacity = np.asarray(base.count_profile.marginal_rates_cmy) * np.asarray(
        base.count_profile.mark_optical_density_cmy
    )
    multiplier = float(
        np.max(maximum_density / base_capacity) * compiler["headroom_ratio"]
    )
    if (
        not math.isfinite(multiplier)
        or multiplier > compiler["maximum_common_rate_multiplier"]
    ):
        raise RuntimeError("required cloud rate multiplier exceeds the frozen limit")
    count = base.count_profile
    compiled_count = CrossLayerPoissonProfile(
        tuple(multiplier * np.asarray(count.marginal_rates_cmy)),
        multiplier * count.shared_all_rate,
        tuple(multiplier * np.asarray(count.shared_pair_rates_cm_cy_my)),
        count.mark_optical_density_cmy,
        0,
        count.component_seed_stride,
    )
    compiled = CrossLayerCloudReferenceProfile(
        compiled_count,
        base.gaussian_sigma_pixels_cmy,
        base.gaussian_truncate,
        base.aperture_factors,
        base.parent_evidence_sha256,
    )
    facts = {
        "maximum_sensitometry_density_cmy": maximum_density.tolist(),
        "base_capacity_cmy": base_capacity.tolist(),
        "common_rate_multiplier": multiplier,
        "compiled_capacity_cmy": (
            np.asarray(compiled_count.marginal_rates_cmy)
            * np.asarray(compiled_count.mark_optical_density_cmy)
        ).tolist(),
    }
    return compiled, facts


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    base = _profile(root)
    compiled, facts = compile_profile(root, contract)
    payload = compiled.to_payload()
    replay = CrossLayerCloudReferenceProfile.from_payload(
        json.loads(json.dumps(payload))
    )
    capacity = np.asarray(facts["compiled_capacity_cmy"])
    maximum = np.asarray(facts["maximum_sensitometry_density_cmy"])
    gates = {
        "capacity": bool(np.all(capacity >= maximum)),
        "correlation_identity": bool(
            np.array_equal(
                base.count_profile.analytic_correlation(),
                compiled.count_profile.analytic_correlation(),
            )
        ),
        "kernel_identity": payload["aperture_kernel_sha256_cmy"]
        == base.to_payload()["aperture_kernel_sha256_cmy"],
        "roundtrip": replay.to_payload() == payload
        and replay.identity() == compiled.identity(),
        "product_disabled": compiled.product_enabled is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "base_profile_identity": base.identity(),
        "compiled_profile_identity": compiled.identity(),
        "compiled_profile_payload_sha256": hashlib.sha256(
            _canonical(payload)
        ).hexdigest(),
        **facts,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dh_sensitometry_cloud_capacity_report.v1",
        "automatic_pass": all(gates.values()),
        "compiled_profile": payload,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["compile_profile", "evaluate", "load_contract"]
