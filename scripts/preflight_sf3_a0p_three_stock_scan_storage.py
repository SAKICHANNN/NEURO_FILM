#!/usr/bin/env python3
"""Check the fixed SF3 colour-baseline scan tier against current P capacity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_scan_storage_preflight import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a0p_three_stock_scan_storage_preflight_v1.json",
    )
    args = parser.parse_args()
    report = evaluate(args.contract, root=ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
