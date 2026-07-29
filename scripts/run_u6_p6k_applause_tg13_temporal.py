#!/usr/bin/env python3
"""Acquire and evaluate the frozen APPLAUSE TG13 temporal confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.applause_tg13_temporal import (  # noqa: E402
    acquire_rows,
    build_report,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6k_applause_tg13_temporal_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p6k_applause_tg13_temporal_v1/report.json",
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire any missing frozen rows before evaluation.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.acquire:
        acquire_rows(ROOT, config)
    report = build_report(ROOT, config)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
