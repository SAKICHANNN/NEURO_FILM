#!/usr/bin/env python
"""Run the frozen U6.P4AW Kodak 250D compatibility experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_250d_same_sheet_granularity import (
    evaluate_compatibility,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p4aw_kodak_250d_same_sheet_granularity_compatibility_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate_compatibility(contract, ROOT)
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"decision={report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
