#!/usr/bin/env python3
"""Run the frozen SF3.A0Y paired shaped-LUT candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.ntire_paired_shaped_lut_candidate import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf3_a0y_ntire_paired_shaped_lut_candidate2_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-row-order", action="store_true")
    parser.add_argument("--cache-root-override", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.config)
    model_lock_path = args.output.with_suffix(".model_lock.json")
    digest = write_report(
        evaluate(
            contract,
            ROOT,
            model_lock_path=model_lock_path,
            reverse_row_order=args.reverse_row_order,
            cache_root_override=args.cache_root_override,
        ),
        args.output,
    )
    print(digest)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
