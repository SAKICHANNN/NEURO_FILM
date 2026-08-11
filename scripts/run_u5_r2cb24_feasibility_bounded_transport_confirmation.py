#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.feasibility_bounded_transport_confirmation import (
    evaluate,
    evaluate_preflight,
    load_contract,
)
from src.eval.safe_base_ao6_chroma_direction import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2cb24_feasibility_bounded_transport_confirmation_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = (
        evaluate_preflight(contract, ROOT)
        if args.preflight_only
        else evaluate(contract, ROOT, args.output_dir)
    )
    digest = write_report(report, args.report)
    print(f"pass={report.get('automatic_pass', report.get('passed'))}")
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
