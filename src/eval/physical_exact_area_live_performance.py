"""U6.P4L fresh-process performance gates for exact-area streaming."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "neuro_film.u6_p4l_exact_area_live_performance_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4l_exact_area_live_performance_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4L parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4L contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["worker"]["network_allowed"]
        or contract["worker"]["gpu_allowed"]
    ):
        raise ValueError("unsupported U6.P4L contract")
    parents = contract["parents"]
    p4k = _load_exact(
        root,
        parents["p4k_contract"],
        parents["p4k_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4k_decision"],
        parents["p4k_decision_sha256"],
    )
    if (
        decision["decision"]
        != "retain_exact_area_row_streaming_open_live_performance_audit"
        or not decision["next_leaf"].startswith("U6.P4L")
    ):
        raise ValueError("U6.P4K did not open live performance audit")
    return contract, p4k


def evaluate_live_performance_records(
    contract: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    worker = contract["worker"]
    gates = contract["automatic_gates"]
    expected_runs = int(worker["fresh_process_runs"])
    if len(runs) != expected_runs:
        raise ValueError("U6.P4L run count does not match the contract")
    hashes = [str(row["stream_sha256"]) for row in runs]
    checks = {
        "stream_hash_exact": all(
            value == worker["expected_stream_sha256"] for value in hashes
        ),
        "repeat_hash_exact": len(set(hashes)) == 1,
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
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
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
    "evaluate_live_performance_records",
    "load_contract",
]
