#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.analytic_y_chromaticity_gold_stress import evaluate, load_contract
from src.eval.monotone_fraction_gold_stress import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2cb52_analytic_y_chromaticity_gold_stress_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.contract), ROOT, args.output_dir)
    digest = write_report(report, args.report)
    print(
        f"automatic_pass={report['automatic_pass']}\ndecision={report['decision']}\nstable_evidence_id={report['stable_evidence_id']}\nreport_sha256={digest}"
    )


if __name__ == "__main__":
    main()
