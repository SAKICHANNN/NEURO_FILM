"""Run the frozen U6.P4HN source-observable copula calibration evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.source_observable_copula_calibration import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(args.contract.read_text("utf-8"))
    report = evaluate(
        root=root,
        contract=contract,
        output_dir=args.output_dir,
        clang=args.clang,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
