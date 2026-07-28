#!/usr/bin/env python
"""Adjudicate two frozen U5.R2AH1D development reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--first-report", type=Path, required=True)
    parser.add_argument("--second-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    first_bytes = args.first_report.read_bytes()
    second_bytes = args.second_report.read_bytes()
    first = json.loads(first_bytes)
    second = json.loads(second_bytes)
    if first["experiment_id"] != config["experiment_id"]:
        raise ValueError("first report experiment identity mismatch")
    if second["experiment_id"] != first["experiment_id"]:
        raise ValueError("second report experiment identity mismatch")
    if first["config_sha256"] != _sha256(config_bytes):
        raise ValueError("first report config identity mismatch")
    if second["config_sha256"] != first["config_sha256"]:
        raise ValueError("second report config identity mismatch")

    reports_byte_identical = first_bytes == second_bytes
    if not reports_byte_identical:
        decision = "close_two_run_nondeterminism"
    elif first["all_development_checks_passed"]:
        decision = "open_untouched_different_family_confirmation"
    else:
        decision = "close_frozen_development_gate_failure"

    output = {
        "schema_version": 1,
        "experiment_id": first["experiment_id"],
        "decision": decision,
        "software_commit": first["software_commit"],
        "config_file_sha256": _sha256(config_bytes),
        "run_a_report_sha256": _sha256(first_bytes),
        "run_b_report_sha256": _sha256(second_bytes),
        "reports_byte_identical": reports_byte_identical,
        "all_development_gates_passed": first[
            "all_development_checks_passed"
        ],
        "checks": first["checks"],
        "failed_checks": sorted(
            name for name, passed in first["checks"].items() if not passed
        ),
        "metrics": first["metrics"],
        "parameter_count": first["parameter_count"],
        "parameter_state_sha256": first["parameter_state_sha256"],
        "training_population_sha256": first["training_population_sha256"],
        "development_population_sha256": first[
            "development_population_sha256"
        ],
        "reserved_confirmation_accessed": first[
            "reserved_confirmation_accessed"
        ],
        "w1_reserved_confirmation_accessed": first[
            "w1_reserved_confirmation_accessed"
        ],
        "branch": (
            "Close without width, depth, training-step, loss, adversary, seed, "
            "threshold, calibration, or post-hoc strength rescue. Retain O0 "
            "independently; do not access either reserved confirmation."
            if decision.startswith("close_")
            else (
                "Freeze the final checkpoint and open only the preregistered "
                "different-family AH1C confirmation."
            )
        ),
        "claim_ceiling": first["claim_ceiling"],
    }
    encoded = (
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "decision": decision,
                "failed_checks": output["failed_checks"],
                "output": str(args.output),
                "sha256": _sha256(encoded),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
