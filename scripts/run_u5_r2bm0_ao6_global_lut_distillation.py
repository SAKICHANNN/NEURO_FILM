#!/usr/bin/env python3
"""Run the frozen BM0 AO6-to-global-LUT grouped development audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_global_lut_distillation import run_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2bm0_ao6_global_lut_distillation_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    software_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = run_audit(
        root=ROOT,
        config=config,
        config_path=config_path,
        output_dir=args.output_dir.resolve(),
        software_commit=software_commit,
    )
    print(json.dumps({
        "stable_evidence_id": report["stable_evidence_id"],
        "automatic_pass": report["automatic_pass"],
        "aggregate": report["aggregate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
