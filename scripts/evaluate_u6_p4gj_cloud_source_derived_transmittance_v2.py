#!/usr/bin/env python3
"""Record the frozen P4GJ source-derived transmittance boundary result."""

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
        raise RuntimeError("P4GJ parent drift")
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
        dll = _build(
            ROOT,
            Path(directory) / "build",
            None,
            bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
        )
        library = _configure_source_derived(dll)
        for level in (0.0, 1.0):
            source = np.full(
                (fixture["height"], fixture["width"], 3),
                level,
                dtype=np.float32,
            )
            error = None
            try:
                render_physical_partition(
                    library,
                    physical,
                    source,
                    0,
                    fixture["height"],
                    source_derived_expected=True,
                )
            except RuntimeError as caught:
                error = str(caught)
            rows.append({"source_level": level, "error": error})
        abi = int(library.nf_sensitometry_cloud_bridge_f32_abi_version_v2())
    checks = {
        "abi_v2": abi == 2,
        "black_endpoint_reaches_bridge": rows[0]["error"] == "P4FB physical provider post failed: 23",
        "full_chain_accepts_endpoints": all(row["error"] is None for row in rows),
        "response_span": False,
        "cross_compiler": False,
        "partition_repeat": False,
        "output_atomic": False,
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "bridge_source_sha256": _sha(
            ROOT / "native/film_physics/nf_sensitometry_cloud_bridge_f32_v2.c"
        ),
        "bridge_header_sha256": _sha(
            ROOT / "native/film_physics/nf_sensitometry_cloud_bridge_f32_v2.h"
        ),
        "rows": rows,
        "gates": checks,
        "decision": contract["decision_if_fail"],
        "failure_mechanism": "source-derived cloud density leaves the physical black/white envelope before bounded adjacency",
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
