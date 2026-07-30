#!/usr/bin/env python3
"""Adjudicate the frozen U5.R2BH1 two-arm blind confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fixed_global_policy_adjudication import adjudicate_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_dir = args.render_dir.resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    payload = adjudicate_files(
        config_path=ROOT
        / "configs/u5_r2bh1_fixed_global_policy_confirmation_v1.json",
        observations_path=ROOT
        / "configs/u5_r2bh1_fixed_global_policy_confirmation_observations_v1.json",
        full_resolution_review_path=ROOT
        / "configs/u5_r2bh1_fixed_global_policy_full_resolution_review_v1.json",
        mapping_receipt_path=ROOT
        / "configs/u5_r2bh1_fixed_global_policy_mapping_receipt_v1.json",
        render_report_path=render_dir / "report.json",
        mapping_paths=[
            render_dir / "blind" / f"blind_round_{index}_mapping.json"
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
                "stable_evidence_id": payload["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
