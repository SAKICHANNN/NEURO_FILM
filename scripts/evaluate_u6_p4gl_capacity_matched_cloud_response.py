#!/usr/bin/env python3
"""Evaluate the P4GL capacity-matched uniform-field cloud response."""

from __future__ import annotations

import argparse
import _ctypes
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
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for parent in contract["parents"].values():
        path = ROOT / parent["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        decision = evidence.get("decision", evidence.get("stable", {}).get("decision"))
        if _sha(path) != parent["sha256"] or decision != parent["required_decision"]:
            raise RuntimeError("P4GL parent drift")
    capacity = evaluate_capacity(
        ROOT, ROOT / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    required_identity = contract["parents"]["capacity_profile"]["required_profile_identity"]
    if profile.identity() != required_identity:
        raise RuntimeError("P4GL compiled cloud profile drift")

    fixture = contract["fixture"]
    physical = json.loads(
        (ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text(
            encoding="utf-8"
        )
    )
    physical["fixture"]["full_height"] = fixture["height"]
    physical["fixture"]["width"] = fixture["width"]
    levels = np.linspace(0.0, 1.0, fixture["build_levels"], dtype=np.float64)
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    all_levels = np.sort(np.concatenate((levels, midpoints)))
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
        try:
            for level in all_levels:
                source = np.full(
                    (fixture["height"], fixture["width"], 3), level, dtype=np.float32
                )
                diagnostics: list[dict[str, float]] = []
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
                    cloud_profile=profile,
                )
                rows.append(
                    {
                        "source_level": float(level),
                        "scan_median_rgb": np.median(output, axis=(0, 1)).tolist(),
                        "output_sha256": hashlib.sha256(memoryview(output).cast("B")).hexdigest(),
                        "density_envelope": diagnostics[0],
                        "new_boundary_fraction": diagnostics[0]["new_boundary_fraction"],
                    }
                )
            probe_level = 0.5
            probe = np.full(
                (fixture["height"], fixture["width"], 3), probe_level, dtype=np.float32
            )
            parts = []
            for start in range(0, fixture["height"], fixture["tile_rows"][0]):
                height = min(fixture["tile_rows"][0], fixture["height"] - start)
                parts.append(
                    render_physical_partition(
                        library,
                        physical,
                        probe,
                        start,
                        height,
                        source_derived_expected=True,
                        enforce_density_envelope=True,
                        exact_sensitometry_endpoints=True,
                        cloud_profile=profile,
                    )
                )
            partition = np.concatenate(parts, axis=0)
            direct = render_physical_partition(
                library,
                physical,
                probe,
                0,
                fixture["height"],
                source_derived_expected=True,
                enforce_density_envelope=True,
                exact_sensitometry_endpoints=True,
                cloud_profile=profile,
            )
        finally:
            _ctypes.FreeLibrary(library._handle)

    medians = np.asarray([row["scan_median_rgb"] for row in rows])
    steps = np.diff(medians, axis=0)
    spans = medians[-1] - medians[0]
    interior = rows[1:-1]
    gates = contract["gates"]
    checks = {
        "endpoint_acceptance": len(rows) == fixture["build_levels"] + fixture["confirmation_midpoints"],
        "response_span": bool(np.min(spans) >= gates["minimum_channel_scan_response_span"]),
        "strict_monotonicity": int(np.sum(steps <= 0.0)) <= gates["maximum_nonpositive_scan_steps"],
        "interior_residual": min(
            row["density_envelope"]["bounded_residual_rms"] for row in interior
        ) >= gates["minimum_interior_bounded_density_residual_rms"],
        "mean_transmittance": max(
            row["density_envelope"]["maximum_absolute_mean_bounded_transmittance_bias"]
            for row in rows
        ) <= gates["maximum_absolute_mean_transmittance_bias"],
        "density_envelope": max(
            row["density_envelope"]["maximum_density_envelope_violation"] for row in rows
        ) == gates["maximum_density_envelope_violation"],
        "boundary": max(row["new_boundary_fraction"] for row in rows)
        <= gates["maximum_new_boundary_fraction"],
        "partition": bool(np.array_equal(partition, direct)),
        "no_hard_clipping": all(
            row["density_envelope"]["hard_clipping_used"] == 0.0 for row in rows
        ),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "compiled_profile_identity": profile.identity(),
        "rows": rows,
        "scan_response_span_rgb": spans.tolist(),
        "nonpositive_scan_steps": int(np.sum(steps <= 0.0)),
        "minimum_scan_step": float(np.min(steps)),
        "partition_probe_sha256": hashlib.sha256(memoryview(direct).cast("B")).hexdigest(),
        "gates": checks,
        "decision": contract["decision_if_pass"] if all(checks.values()) else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("-contract", "-result"),
        "automatic_pass": all(checks.values()),
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
