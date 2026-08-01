#!/usr/bin/env python
"""Run the frozen FiveK explicit-operator style-capacity diagnostic."""

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

from src.eval.fivek_operator_style_ceiling_run import (  # noqa: E402
    run_operator_style_ceiling,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bq4d_fivek_operator_style_ceiling_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = run_operator_style_ceiling(
        root=ROOT,
        config=config,
        config_path=args.config,
        output_path=args.output,
        software_commit=commit,
    )
    print(
        json.dumps(
            {
                "branch": report["branch"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
