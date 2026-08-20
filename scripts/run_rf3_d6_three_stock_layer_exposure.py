#!/usr/bin/env python3
"""Run the frozen RF3.D6 layer-exposure observability experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_layer_exposure import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/rf3_d6_three_stock_layer_exposure_observability_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/rf3_d6_three_stock_layer_exposure_observability_v1")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    hashes = []
    for name in ("run_a", "run_b"):
        report = evaluate(contract, ROOT)
        hashes.append(write_report(report, args.output_dir / name / "report.json"))
    if len(set(hashes)) != 1:
        raise RuntimeError("RF3.D6 formal reports differ")
    print(json.dumps({"report_sha256": hashes[0], "report": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
