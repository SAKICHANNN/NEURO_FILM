#!/usr/bin/env python3
"""Run fixed BL5 strict-interior OOD evaluation and optional blind sheets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_strict_interior_ood import (  # noqa: E402
    build_bl6_blind_round,
    run_strict_interior_ood,
    validate_bl6_contract,
)


CONFIG = ROOT / "configs/u5_r2bl6_filmmatch_strict_interior_ood_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-blind-sheets", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_dir = ROOT / config["population"]["bl3_parent_output_dir"]
    report = run_strict_interior_ood(
        root=ROOT,
        config=config,
        config_path=CONFIG,
        parent_output_dir=parent_dir,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    mappings = []
    if args.build_blind_sheets:
        if not report["visual_review_opened"]:
            raise RuntimeError("automatic gate forbids blind sheets")
        validated = validate_bl6_contract(ROOT, config)
        for round_index in (1, 2, 3):
            mappings.append(
                build_bl6_blind_round(
                    root=ROOT,
                    parent_output_dir=parent_dir,
                    candidate_output_dir=output_dir,
                    source_rows=validated["source_rows"],
                    eligible_ids=validated["eligible_ids"],
                    round_index=round_index,
                    output_dir=output_dir / "blind",
                )
            )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "aggregate": report["aggregate"],
                "stable_evidence_id": report["stable_evidence_id"],
                "mapping_sha256": [row["mapping_sha256"] for row in mappings],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
