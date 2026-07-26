#!/usr/bin/env python
"""Run the frozen U5.R2L1 full-frame strength preflight audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.full_frame_strength_preflight import (  # noqa: E402
    evaluate_full_frame_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2l1_full_frame_strength_preflight_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2l1_full_frame_strength_preflight_v1/report.json"
        ),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    config_bytes = config_path.read_bytes()
    report = evaluate_full_frame_preflight(
        root=ROOT,
        config=json.loads(config_bytes),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "all_checks_passed": report["all_checks_passed"],
                "counts": report["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
