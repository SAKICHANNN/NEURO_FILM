"""U6.P4IH analytical shared-direction dye-amount envelope."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.bounded_dye_amount_direction import (
    apply_bounded_dye_amount_direction,
)

SCHEMA = "neuro-film.u6-p4ih-bounded-dye-amount-direction-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ih-bounded-dye-amount-direction-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IH contract")
    return payload


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    binding = config["parent"]
    parent_path = root / str(binding["path"])
    if not parent_path.is_file() or _hash(parent_path) != binding["sha256"]:
        raise ValueError("P4IH parent integrity mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("decision") != binding["required_decision"]
        or parent.get("formal_runs", {}).get("stable_evidence_id")
        != binding["required_stable_evidence_id"]
    ):
        raise ValueError("P4IH parent decision drift")
    levels = np.linspace(0.0, 1.0, int(config["population"]["levels"]), dtype=np.float64)
    base = np.asarray([(a, b, c) for a in levels for b in levels for c in levels])
    direction_errors: list[float] = []
    limited_fractions: list[float] = []
    residual_rms: list[float] = []
    output_hashes: list[str] = []
    minimum = 1.0
    maximum = 0.0
    for direction in config["population"]["residual_directions"]:
        vector = np.asarray(direction, dtype=np.float64)
        for amplitude in config["population"]["residual_amplitudes"]:
            candidate = base + float(amplitude) * vector
            output, receipt = apply_bounded_dye_amount_direction(base, candidate)
            direction_errors.append(receipt["maximum_shared_direction_error"])
            limited_fractions.append(receipt["limited_fraction"])
            residual_rms.append(receipt["bounded_residual_rms"])
            output_hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
            minimum = min(minimum, float(np.min(output)))
            maximum = max(maximum, float(np.max(output)))
    metrics = {
        "base_sample_count": len(base),
        "case_count": len(output_hashes),
        "maximum_shared_direction_error": max(direction_errors),
        "minimum_output": minimum,
        "maximum_output": maximum,
        "minimum_limited_fraction": min(limited_fractions),
        "maximum_limited_fraction": max(limited_fractions),
        "minimum_bounded_residual_rms": min(residual_rms),
        "output_inventory_sha256": hashlib.sha256("".join(output_hashes).encode()).hexdigest(),
    }
    gates = config["gates"]
    checks = {
        "direction": metrics["maximum_shared_direction_error"] <= gates["maximum_shared_direction_error"],
        "bounded": metrics["minimum_output"] >= -gates["maximum_output_boundary_error"]
        and metrics["maximum_output"] <= 1.0 + gates["maximum_output_boundary_error"],
        "limited_cases_present": metrics["maximum_limited_fraction"] >= gates["minimum_limited_fraction"],
        "unlimited_cases_present": metrics["minimum_limited_fraction"] <= 1.0 - gates["minimum_unlimited_fraction"],
        "material": metrics["minimum_bounded_residual_rms"] >= gates["minimum_bounded_residual_rms"],
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["evaluate", "load_contract", "write_report"]
