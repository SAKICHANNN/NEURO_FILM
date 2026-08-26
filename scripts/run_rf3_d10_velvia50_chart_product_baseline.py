#!/usr/bin/env python3
"""Run the frozen RF3.D10 Velvia chart product baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.velvia50_chart_product_baseline import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/rf3_d10_velvia50_chart_product_baseline_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = evaluate(root=ROOT, contract_path=args.contract, reverse=args.reverse)
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps(report["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
