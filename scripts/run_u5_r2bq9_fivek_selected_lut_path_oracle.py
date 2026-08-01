#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.fivek_selected_lut_path_oracle_run import run_selected_lut_path_oracle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2bq9_fivek_selected_lut_path_oracle_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = run_selected_lut_path_oracle(root=ROOT, config=config, config_path=args.config, output_path=args.output, software_commit=commit)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
