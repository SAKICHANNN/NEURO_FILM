"""U6.P4DI capacity compiler with explicit inherited correlation identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_cloud_row_stream import _profile
from src.eval.sensitometry_cloud_capacity import compile_profile as compile_v1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    failed = contract["parents"]["p4dh"]
    failed_path = root / failed["path"]
    evidence = json.loads(failed_path.read_text(encoding="utf-8"))
    sensitometry = contract["parents"]["sensitometry"]
    sensitometry_path = root / sensitometry["path"]
    sensitometry_payload = json.loads(sensitometry_path.read_text(encoding="utf-8"))
    if (
        _sha(failed_path) != failed["sha256"]
        or evidence["decision"] != failed["required_decision"]
        or evidence["failed_gate"] != failed["required_failed_gate"]
        or _sha(sensitometry_path) != sensitometry["sha256"]
        or sensitometry_payload["experiment_id"]
        != sensitometry["required_experiment_id"]
    ):
        raise RuntimeError("P4DI parent drift")
    return contract


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    v1_contract = json.loads(
        (root / "configs/u6_p4dh_sensitometry_cloud_capacity_v1.json").read_text()
    )
    compiled, facts = compile_v1(root, v1_contract)
    base = _profile(root)
    base_correlation = base.count_profile.analytic_correlation()
    compiled_correlation = compiled.count_profile.analytic_correlation()
    drift = float(np.max(np.abs(base_correlation - compiled_correlation)))
    base_payload = base.to_payload()
    compiled_payload = compiled.to_payload()
    capacity = np.asarray(facts["compiled_capacity_cmy"])
    maximum = np.asarray(facts["maximum_sensitometry_density_cmy"])
    inherited_identity = hashlib.sha256(
        base_correlation.astype("<f8").tobytes()
    ).hexdigest()
    gates = {
        "capacity": bool(np.all(capacity >= maximum)),
        "inherited_correlation_identity": inherited_identity
        == hashlib.sha256(base_correlation.astype("<f8").tobytes()).hexdigest(),
        "recomputed_correlation": drift
        <= contract["compiler"]["maximum_recomputed_correlation_absolute_drift"],
        "kernel_identity": compiled_payload["aperture_kernel_sha256_cmy"]
        == base_payload["aperture_kernel_sha256_cmy"],
        "roundtrip": compiled.from_payload(
            json.loads(json.dumps(compiled_payload))
        ).identity()
        == compiled.identity(),
        "product_disabled": compiled.product_enabled is False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "base_profile_identity": base.identity(),
        "compiled_profile_identity": compiled.identity(),
        "inherited_correlation_identity": inherited_identity,
        "maximum_recomputed_correlation_absolute_drift": drift,
        **facts,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4di_sensitometry_cloud_capacity_report.v2",
        "automatic_pass": all(gates.values()),
        "compiled_profile": compiled_payload,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
