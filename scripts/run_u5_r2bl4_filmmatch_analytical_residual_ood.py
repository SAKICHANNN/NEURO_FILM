#!/usr/bin/env python3
"""Run the frozen BL4 analytical safe-residual composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_analytical_residual_ood import (  # noqa: E402
    run_analytical_residual_ood,
)


CONFIG = ROOT / "configs/u5_r2bl4_filmmatch_analytical_residual_ood_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2bl3_filmmatch_identity_residual_ood_v1/run_a",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    parent_dir = (
        args.parent_output_dir
        if args.parent_output_dir.is_absolute()
        else ROOT / args.parent_output_dir
    )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = run_analytical_residual_ood(
        root=ROOT,
        config=config,
        config_path=CONFIG,
        parent_output_dir=parent_dir,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "aggregate": report["aggregate"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
