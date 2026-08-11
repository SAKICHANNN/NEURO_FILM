#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.analytic_y_chromaticity_adjudication import adjudicate, load_contract
from src.eval.safe_base_ao6_chroma_direction import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2cb50_analytic_y_chromaticity_adjudication_v1.json",
    )
    parser.add_argument("--automatic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(load_contract(args.contract), ROOT, args.automatic_report)
    digest = write_report(result, args.output)
    print(
        f"pass={result['pass']}\ndecision={result['decision']}\nstable_evidence_id={result['stable_evidence_id']}\nreport_sha256={digest}"
    )


if __name__ == "__main__":
    main()
