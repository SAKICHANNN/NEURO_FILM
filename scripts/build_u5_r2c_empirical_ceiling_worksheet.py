#!/usr/bin/env python3
"""Build the frozen U5.R2C non-binding annotation-workload worksheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.empirical_ceiling import (  # noqa: E402
    annotation_workload,
    load_empirical_ceiling_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2c_empirical_ceiling_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "u5_r2c_empirical_ceiling_v1" / "worksheet.json",
    )
    args = parser.parse_args()

    config = load_empirical_ceiling_contract(args.config)
    scenarios = []
    for scene_count in config["workload_scenarios_n"]:
        scenarios.append(
            {
                "base": annotation_workload(config, scene_count=scene_count),
                "all_severity_escalates": annotation_workload(
                    config,
                    scene_count=scene_count,
                    candidate_severity_escalation_fraction=1.0,
                    deployed_severity_escalation_fraction=1.0,
                ),
            }
        )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "candidate_budget_k": config["candidate_budget_k"],
        "identity_fallback_outside_k": True,
        "global_comparator": config["global_comparator"],
        "selection_and_evaluation_panels_must_be_disjoint": True,
        "all_scenes_remain_in_every_denominator": True,
        "workload_scenarios": scenarios,
        "binding_estimator": None,
        "binding_sample_size": None,
        "human_recruitment_authorized": False,
        "pixel_acquisition_authorized": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "scenario_count": len(scenarios),
                "binding_sample_size": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
