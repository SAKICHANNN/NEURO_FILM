"""U6.P4Q explicit scalar-CDF integration and 24MP audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.physical_stationary_global_live_performance import (
    compiled_profiles,
    stationary_target,
    stream_hash,
)


SCHEMA = (
    "neuro_film.u6_p4q_constant_rate_integration_live_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u6_p4q_constant_rate_integration_live_report.v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4Q parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4Q contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    executor = contract["executor"]
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or executor["mode"] != "scalar-cdf-v1"
        or executor["legacy_default_mode"] != "legacy-v1"
        or executor["nonconstant_rate_behavior"] != "fail-closed"
        or executor["full_output_assembly_allowed"]
        or executor["full_target_allocation_allowed"]
        or executor["network_allowed"]
        or executor["gpu_allowed"]
    ):
        raise ValueError("unsupported U6.P4Q contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    _load_exact(
        root,
        parents["p4o_contract"],
        parents["p4o_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4p_decision"],
        parents["p4p_decision_sha256"],
    )
    if (
        decision["decision"]
        != "retain_exact_constant_rate_executor_open_versioned_integration"
        or not decision["next_leaf"].startswith("U6.P4Q")
    ):
        raise ValueError("U6.P4P did not open versioned integration")
    return contract, parent


def small_integration_evidence(
    contract: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    executor = contract["executor"]
    target = stationary_target(
        tuple(int(value) for value in executor["small_parity_shape"]),
        executor["target_density"],
    )
    profiles = compiled_profiles(contract, parent)
    partitions = {}
    for height in executor["partition_test_heights"]:
        key = str(int(height))
        legacy = stream_hash(
            target,
            profiles,
            row_tile_height=int(height),
            executor_mode="legacy-v1",
        )
        scalar = stream_hash(
            target,
            profiles,
            row_tile_height=int(height),
            executor_mode="scalar-cdf-v1",
        )
        partitions[key] = {
            "legacy_sha256": legacy["stream_sha256"],
            "scalar_sha256": scalar["stream_sha256"],
            "legacy_expected": (
                legacy["stream_sha256"]
                == executor["expected_legacy_stream_sha256"][key]
            ),
            "scalar_legacy_exact": (
                scalar["stream_sha256"] == legacy["stream_sha256"]
            ),
        }
    nonconstant = target.copy()
    nonconstant[0, 0, 0] += 0.01
    rejected = False
    try:
        stream_hash(
            nonconstant,
            profiles,
            row_tile_height=31,
            executor_mode="scalar-cdf-v1",
        )
    except ValueError:
        rejected = True
    return {
        "partitions": partitions,
        "nonconstant_scalar_request_rejected": rejected,
    }


def evaluate_live_records(
    contract: dict[str, Any],
    small: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = contract["automatic_gates"]
    expected = int(contract["executor"]["fresh_process_runs"])
    if len(runs) != expected:
        raise ValueError("U6.P4Q run count does not match contract")
    hashes = [str(row.get("stream_sha256", "")) for row in runs]
    checks = {
        "legacy_default_hash_unchanged": all(
            row["legacy_expected"]
            for row in small["partitions"].values()
        )
        == bool(gates["legacy_default_hash_unchanged"]),
        "scalar_legacy_small_byte_exact": all(
            row["scalar_legacy_exact"]
            for row in small["partitions"].values()
        )
        == bool(gates["scalar_legacy_small_byte_exact"]),
        "nonconstant_scalar_request_rejected": bool(
            small["nonconstant_scalar_request_rejected"]
        )
        == bool(gates["nonconstant_scalar_request_rejected"]),
        "repeat_stream_hash_exact": len(set(hashes)) == 1
        and bool(hashes[0])
        == bool(gates["repeat_stream_hash_exact"]),
        "output_rows": all(
            int(row.get("output_rows", 0)) == int(gates["output_rows"])
            for row in runs
        ),
        "process_tree_peak_rss": max(
            int(row["process_tree_peak_rss_bytes"]) for row in runs
        )
        <= int(gates["maximum_process_tree_peak_rss_bytes"]),
        "worker_wall": max(float(row["wall_seconds"]) for row in runs)
        <= float(gates["maximum_worker_wall_seconds"]),
        "liveness_samples": min(
            int(row["liveness_samples"]) for row in runs
        )
        >= int(gates["minimum_liveness_samples"]),
        "exit_code": all(
            int(row["exit_code"]) == int(gates["exit_code"])
            for row in runs
        ),
        "stderr_empty": all(
            int(row["stderr_bytes"]) == 0 for row in runs
        )
        == bool(gates["stderr_empty"]),
        "owned_temp_cleanup": sum(
            int(row["owned_temp_residue_count"]) for row in runs
        )
        == int(gates["owned_temp_residue_count"]),
        "physical_domain": min(
            float(row.get("minimum_density", -1.0)) for row in runs
        )
        >= 0.0
        and max(
            float(row.get("maximum_transmittance", 2.0))
            for row in runs
        )
        <= 1.0,
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "small_integration": small,
        "runs": runs,
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_live_records",
    "load_contract",
    "small_integration_evidence",
]
