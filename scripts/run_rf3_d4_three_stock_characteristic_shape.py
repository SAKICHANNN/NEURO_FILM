#!/usr/bin/env python3
"""Run the frozen RF3.D4 characteristic-shape discriminator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_characteristic_shape import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "rf3_d4_three_stock_characteristic_shape_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.contract), ROOT, overlay_dir=args.overlay_dir)
    digest = write_report(report, args.output)
    print(json.dumps({"report_sha256": digest, **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
