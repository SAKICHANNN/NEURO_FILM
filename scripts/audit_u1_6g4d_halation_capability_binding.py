"""Audit concrete U1.6G4D provider bindings and static G3 readiness."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import (  # noqa: E402
    REQUIRED_INTEGRATION_CAPABILITIES,
    available_halation_integration_capabilities,
    build_halation_resource_plan,
    resolve_halation_integration_capability_bindings,
)

RESOURCE_FIELDS = (
    "blur_count",
    "percentile_count",
    "finite_halo_blurs",
    "global_grid_blurs",
    "external_bytes",
    "required_output_bytes",
    "peak_context_bytes",
    "peak_scalar_bytes",
    "peak_workspace_bytes",
    "peak_planner_ram_bytes",
    "scratch_disk_bytes",
    "maximum_output_input_halo",
)


def run(config: dict) -> dict:
    bindings = resolve_halation_integration_capability_bindings()
    binding_rows = [dataclasses.asdict(binding) for binding in bindings]
    binding_match = binding_rows == config["bindings"]
    capabilities = available_halation_integration_capabilities()
    capability_match = capabilities == REQUIRED_INTEGRATION_CAPABILITIES
    cases = []
    for case in config["cases"]:
        shape = tuple(int(item) for item in case["source_shape"])
        for family in config["families"]:
            unbound = build_halation_resource_plan(
                family,
                shape,
                tile_size=int(case["tile_size"]),
            )
            bound = build_halation_resource_plan(
                family,
                shape,
                tile_size=int(case["tile_size"]),
                available_capabilities=capabilities,
            )
            graph_equal = bound.nodes == unbound.nodes and bound.lifetimes == unbound.lifetimes
            resources_equal = all(
                getattr(bound, field) == getattr(unbound, field) for field in RESOURCE_FIELDS
            )
            cases.append(
                {
                    "family": family,
                    "source_shape": list(shape),
                    "tile_size": int(case["tile_size"]),
                    "unbound_fingerprint": unbound.fingerprint,
                    "bound_fingerprint": bound.fingerprint,
                    "bound_integration_ready": bound.integration_ready,
                    "bound_missing_capabilities": list(bound.missing_capabilities),
                    "bound_unresolved_workspace_nodes": list(bound.unresolved_workspace_nodes),
                    "nodes_and_lifetimes_equal": graph_equal,
                    "resource_fields_equal": resources_equal,
                    "blur_count": bound.blur_count,
                    "percentile_count": bound.percentile_count,
                    "finite_halo_blurs": list(bound.finite_halo_blurs),
                    "global_grid_blurs": list(bound.global_grid_blurs),
                }
            )
    cases_pass = all(
        row["bound_integration_ready"]
        and not row["bound_missing_capabilities"]
        and not row["bound_unresolved_workspace_nodes"]
        and row["nodes_and_lifetimes_equal"]
        and row["resource_fields_equal"]
        for row in cases
    )
    passed = binding_match and capability_match and cases_pass
    return {
        "schema_version": 1,
        "node": config["node"],
        "bindings": binding_rows,
        "resolved_capabilities": sorted(capabilities),
        "cases": cases,
        "gate_result": {
            "exact_provider_identity": binding_match,
            "resolved_capabilities_equal_required": capability_match,
            "bound_plans_integration_ready": cases_pass,
            "unbound_graph_and_resources_unchanged": cases_pass,
            "passed": passed,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
