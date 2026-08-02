#!/usr/bin/env python
"""Run the frozen U6.P4CD Poisson-binomial granularity baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.poisson_binomial_granularity import (
    evaluate_compatibility,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4cd_poisson_binomial_granularity_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_compatibility(load_contract(args.contract), ROOT)
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"decision={report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
