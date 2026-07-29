#!/usr/bin/env python3
"""Run the frozen U6.P8AM native end-to-end conformance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_native_end_to_end_conformance import (
    run_native_end_to_end_conformance,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8am_native_end_to_end_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u6_p8am_native_end_to_end",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_native_end_to_end_conformance(
        root=ROOT,
        config=config,
        output_dir=args.output_dir,
    )
    write_report(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
