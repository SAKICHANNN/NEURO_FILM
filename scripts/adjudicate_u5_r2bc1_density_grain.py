#!/usr/bin/env python3
"""Adjudicate frozen BC1 masked observations after mapping reveal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_density_grain_value import adjudicate_masked_observations
from src.eval.ao6_procedural_filmfx_value import sha256_file

CONFIG = ROOT / "configs" / "u5_r2bc1_ao6_density_grain_value_v1.json"
OBSERVATIONS = (
    ROOT
    / "configs"
    / "u5_r2bc1_ao6_density_grain_visual_observations_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    observations = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    mapping_path = run_dir / "randomized_blind" / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    result = adjudicate_masked_observations(
        config=config,
        observations=observations,
        mapping=mapping,
    )
    report_path = run_dir / "report.json"
    decision = {
        "schema": "neuro-film.u5-r2bc1-density-grain-decision.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(CONFIG),
        "report_sha256": sha256_file(report_path),
        "observations_sha256": sha256_file(OBSERVATIONS),
        "mapping_sha256": sha256_file(mapping_path),
        "automatic": json.loads(report_path.read_text(encoding="utf-8"))[
            "automatic"
        ],
        "visual": result,
        "claim_ceiling": config["claim_ceiling"],
    }
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
