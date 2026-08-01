#!/usr/bin/env python
"""Run the frozen source-only FiveK duplicate adjudication successor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_duplicate_adjudication import run_adjudication  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bq0s3_fivek_duplicate_adjudication_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = run_adjudication(
        root=ROOT,
        config=config,
        config_path=args.config,
        output_path=args.output,
        software_commit=commit,
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
