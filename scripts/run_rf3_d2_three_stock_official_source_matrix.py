#!/usr/bin/env python3
"""Run the RF3.D2 three-stock current first-party source audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.official_three_stock_prior_source import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rf3_d2_three_stock_official_source_matrix_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-acquire", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.config)
    report = evaluate(contract, ROOT, acquire_missing=not args.no_acquire)
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"decision={report['decision']}")
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
