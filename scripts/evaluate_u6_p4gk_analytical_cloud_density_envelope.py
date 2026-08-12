#!/usr/bin/env python3
"""Evaluate the P4GK endpoint preflight for analytical density bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure_source_derived,
    render_physical_partition,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent_path = ROOT / contract["parent"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        _sha(parent_path) != contract["parent"]["sha256"]
        or parent["stable"]["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4GK parent drift")
    fixture = contract["fixture"]
    physical = json.loads(
        (ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text(
            encoding="utf-8"
        )
    )
    physical["fixture"]["full_height"] = fixture["height"]
    physical["fixture"]["width"] = fixture["width"]
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        library = _configure_source_derived(
            _build(
                ROOT,
                Path(directory) / "build",
                None,
                bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
            )
        )
        for level in (0.0, 1.0):
            source = np.full(
                (fixture["height"], fixture["width"], 3),
                level,
                dtype=np.float32,
            )
            diagnostics = []
            output = render_physical_partition(
                library,
                physical,
                source,
                0,
                fixture["height"],
                source_derived_expected=True,
                enforce_density_envelope=True,
                exact_sensitometry_endpoints=True,
                density_envelope_diagnostics=diagnostics,
            )
            rows.append(
                {
                    "source_level": level,
                    "scan_median_rgb": np.median(output, axis=(0, 1)).tolist(),
                    "output_sha256": hashlib.sha256(memoryview(output).cast("B")).hexdigest(),
                    "density_envelope": diagnostics,
                }
            )
    span = (np.asarray(rows[1]["scan_median_rgb"]) - rows[0]["scan_median_rgb"]).tolist()
    gates = contract["gates"]
    checks = {
        "endpoint_acceptance": True,
        "density_envelope": all(
            item["maximum_density_envelope_violation"] == 0.0
            for row in rows
            for item in row["density_envelope"]
        ),
        "response_span": min(span) >= gates["minimum_channel_scan_response_span"],
        "remaining_uniform_fields": False,
        "partition_process_exact": False,
        "interior_residual": False,
        "boundary": False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "rows": rows,
        "scan_response_span_rgb": span,
        "gates": checks,
        "decision": contract["decision_if_fail"],
        "stop_reason": "endpoint response span fails before remaining uniform fields",
        "mechanism_diagnosis": "P4FB particle-count capacity and fixed 1.5 density normalization are not mean preserving",
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("-contract", "-result"),
        "automatic_pass": False,
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = evaluate(arguments.contract)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if arguments.output is None:
        print(raw)
    else:
        arguments.output.write_text(raw, encoding="utf-8")


if __name__ == "__main__":
    main()
