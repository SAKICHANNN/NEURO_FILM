#!/usr/bin/env python3
"""Acquire and audit the frozen U5.R2BK17S sixth-fresh RAW pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rawpixls_confirmation_preflight import run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bk17s_factorized_ao6_sixth_fresh_source_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    result = run_preflight(
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
                "decoded_rows": report["decoded_row_count"],
                "failures": report["failures"],
                "manifest_sha256": result["manifest_sha256"],
                "report_sha256": result["report_sha256"],
                "visual_review_required": report["visual_review_required"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
