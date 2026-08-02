#!/usr/bin/env python3
"""Run the frozen APPLAUSE effective edge-SFR temporal audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.applause_effective_sfr_temporal import (
    build_development_lock,
    build_report_from_lock,
    write_json,
)
from src.eval.applause_tg13_temporal import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6ai_applause_effective_sfr_temporal_v1.json",
    )
    parser.add_argument(
        "--development-lock",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p6ai_applause_effective_sfr_temporal_v1/development_lock.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p6ai_applause_effective_sfr_temporal_v1/report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_sha256 = sha256_file(args.config)
    development_lock = build_development_lock(
        ROOT, config, contract_sha256=contract_sha256
    )
    write_json(args.development_lock, development_lock)
    persisted_lock = json.loads(args.development_lock.read_text(encoding="utf-8"))
    report = build_report_from_lock(
        ROOT,
        config,
        persisted_lock,
        contract_sha256=contract_sha256,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
