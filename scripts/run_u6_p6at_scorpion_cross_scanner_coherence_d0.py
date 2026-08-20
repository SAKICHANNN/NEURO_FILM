"""Acquire or evaluate the frozen U6.P6AT source cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scorpion_cross_scanner_coherence_d0 import (
    acquire,
    evaluate,
    load_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6at_scorpion_cross_scanner_coherence_d0_v1.json",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--acquisition-report", type=Path)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.config)
    if args.acquire:
        result = acquire(contract, args.data_root)
    else:
        if args.acquisition_report is None:
            raise ValueError("--acquisition-report is required for evaluation")
        result = evaluate(contract, args.data_root, args.acquisition_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            result["aggregate"]
            if "aggregate" in result
            else {"outputs": len(result["outputs"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
