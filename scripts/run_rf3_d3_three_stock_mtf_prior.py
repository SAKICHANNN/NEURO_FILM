#!/usr/bin/env python3
"""Run the frozen RF3.D3 three-stock MTF prior comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_mtf_prior import evaluate, load_contract, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rf3_d3_three_stock_mtf_prior_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.config), ROOT, overlay_dir=args.overlay_dir)
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"decision={report['decision']}")
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
