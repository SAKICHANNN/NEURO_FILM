#!/usr/bin/env python3
"""Run the frozen U5.R2BK10 primitive and severe-failure regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.orthogonal_perceptual_residual_regression import (  # noqa: E402
    run_regression,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/u5_r2bk10_orthogonal_perceptual_residual_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    result = run_regression(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        config_path=config_path,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
    )
    print(
        json.dumps(
            {
                **result["report"],
                "report_sha256": result["report_sha256"],
                "visual_sheet": result["visual_sheet"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
