#!/usr/bin/env python3
"""Run the frozen SF3.A3Q YFCC connectivity adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_three_stock_generation_connectivity import (
    run_connectivity_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/sf3_a3q_yfcc_three_stock_generation_connectivity_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-stock-order", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stock_order = list(config["target_stock_ids"])
    if args.reverse_stock_order:
        stock_order.reverse()
    report = run_connectivity_audit(args.config, stock_order=stock_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "qualifying_author_count": report["portra_generation"][
                    "qualifying_current_portra_three_stock_author_uid_count"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
