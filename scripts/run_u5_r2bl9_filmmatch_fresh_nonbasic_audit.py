#!/usr/bin/env python3
"""Run the frozen BL9 best-basic explainability audit."""

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

from src.eval.filmmatch_fresh_nonbasic_audit import run_nonbasic_audit  # noqa: E402


CONFIG = ROOT / "configs/u5_r2bl9_filmmatch_fresh_nonbasic_audit_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = run_nonbasic_audit(
        root=ROOT,
        config=config,
        config_path=CONFIG,
        output_path=output,
        software_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "summaries": report["summaries"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
