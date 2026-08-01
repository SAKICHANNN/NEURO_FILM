#!/usr/bin/env python3
"""Adjudicate the frozen corrected BN3 visual product-value evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_projection_curve_visual_adjudication import (  # noqa: E402
    adjudicate_files,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_a = args.run_a if args.run_a.is_absolute() else ROOT / args.run_a
    run_b = args.run_b if args.run_b.is_absolute() else ROOT / args.run_b
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    payload = adjudicate_files(
        root=ROOT,
        config_path=ROOT
        / "configs/u5_r2bn3_projection_curve_visual_product_value_v2.json",
        observations_path=ROOT
        / "configs/u5_r2bn3_projection_curve_blind_observations_v2.json",
        full_resolution_review_path=ROOT
        / "configs/u5_r2bn3_projection_curve_full_resolution_review_v2.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2bn3_projection_curve_mapping_receipt_v2.json",
        render_report_paths=[run_a / "report.json", run_b / "report.json"],
        mapping_paths=[
            run_a / "blind" / f"blind_round_{index}_mapping.json"
            for index in (1, 2, 3)
        ],
        adjudicator_software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pass": payload["pass"],
                "aggregate_counts": payload["aggregate_counts"],
                "adaptive_round_wins": payload["adaptive_round_wins"],
                "adaptive_source_majorities": payload[
                    "adaptive_source_majorities"
                ],
                "stable_evidence_id": payload["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
