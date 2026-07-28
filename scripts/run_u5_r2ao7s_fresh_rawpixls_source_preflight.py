#!/usr/bin/env python
"""Acquire and audit the frozen AO7S fresh RAW source pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rawpixls_confirmation_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ao7s_fresh_rawpixls_source_preflight_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "outputs/u5_r2ao7s_fresh_rawpixls_source_preflight_v1"
        ),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = run_preflight(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        config_path=config_path,
        output_dir=output_dir,
        software_commit=commit,
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decoded_rows": report["decoded_row_count"],
                "failures": report["failure_count"],
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
