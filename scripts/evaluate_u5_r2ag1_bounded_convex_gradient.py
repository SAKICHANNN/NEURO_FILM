#!/usr/bin/env python
"""Adjudicate the two frozen U5.R2AG1 representation reports."""

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
    first_bytes = args.first_report.read_bytes()
    second_bytes = args.second_report.read_bytes()
    first = json.loads(first_bytes)
    second = json.loads(second_bytes)
    if first["experiment_id"] != json.loads(config_bytes)["experiment_id"]:
        raise ValueError("first report experiment identity mismatch")
    if second["experiment_id"] != first["experiment_id"]:
        raise ValueError("second report experiment identity mismatch")
    reports_byte_identical = first_bytes == second_bytes
    if not reports_byte_identical:
        decision = "close_two_run_nondeterminism"
    elif first["all_checks_passed"]:
        decision = "retain_compact_synthetic_representation"
    else:
        decision = "close_frozen_gate_failure"

    target_summary = {
        name: {
            "confirmation_rgb_rmse": report["confirmation_rgb_rmse"],
            "global_affine_confirmation_rgb_rmse": (
                report["global_affine_confirmation_rgb_rmse"]
            ),
            "improvement_over_global_affine_fraction": (
                report["improvement_over_global_affine_fraction"]
            ),
            "minimum_analytic_jacobian_eigenvalue": (
                report["minimum_analytic_jacobian_eigenvalue"]
            ),
            "minimum_analytic_jacobian_determinant": (
                report["minimum_analytic_jacobian_determinant"]
            ),
            "maximum_analytic_jacobian_spectral_norm": (
                report["maximum_analytic_jacobian_spectral_norm"]
            ),
            "inverse_roundtrip_maximum_absolute_error": (
                report["inverse_roundtrip_maximum_absolute_error"]
            ),
            "partition_maximum_absolute_error": (
                report["partition_maximum_absolute_error"]
            ),
        }
        for name, report in first["targets"].items()
    }
    output = {
        "schema_version": 1,
        "experiment_id": first["experiment_id"],
        "decision": decision,
        "software_commit": first["software_commit"],
        "config_file_sha256": _sha256(config_bytes),
        "run_a_report_sha256": _sha256(first_bytes),
        "run_b_report_sha256": _sha256(second_bytes),
        "reports_byte_identical": reports_byte_identical,
        "checks": first["checks"],
        "all_frozen_gates_passed": first["all_checks_passed"],
        "failed_checks": sorted(
            name for name, passed in first["checks"].items() if not passed
        ),
        "targets": target_summary,
        "branch": (
            "The representation closes without anchor-count, affine-wrapper, "
            "temperature, optimizer, tolerance, or partition-parity rescue."
            if decision.startswith("close_")
            else "Retain only as compact paired-synthetic representation evidence."
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
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "failed_checks": output["failed_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
