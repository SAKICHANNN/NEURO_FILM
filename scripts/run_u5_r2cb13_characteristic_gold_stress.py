#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.characteristic_gold_stress import evaluate, load_contract, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-persist-outputs", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        load_contract(args.contract),
        ROOT,
        args.output_dir,
        persist_outputs=not args.no_persist_outputs,
    )
    digest = write_report(report, args.report)
    print(f"automatic_pass={report['automatic_pass']}")
    print(
        f"failed_gates={','.join(k for k, value in report['checks'].items() if not value)}"
    )
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
