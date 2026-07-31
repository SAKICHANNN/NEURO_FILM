#!/usr/bin/env python3
"""Run the frozen U5.R2BK18 four-arm sixth-fresh confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.factorized_ao6_sixth_fresh_confirmation import run_confirmation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bk18_factorized_ao6_sixth_fresh_confirmation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    result = run_confirmation(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        config_path=config_path,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "automatic_gates": report["automatic_gates"],
                "population_median_style_delta_e76": report[
                    "population_median_style_delta_e76"
                ],
                "bk16_population_median_increment_vs_safe_delta_e76": report[
                    "bk16_population_median_increment_vs_safe_delta_e76"
                ],
                "maximum_new_code_boundary_fraction_vs_source": report[
                    "maximum_new_code_boundary_fraction_vs_source"
                ],
                "report_sha256": result["report_sha256"],
                "stable_evidence_id": report["stable_evidence_id"],
                "visual_sheets": result["visual_sheets"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
