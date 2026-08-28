#!/usr/bin/env python3
"""Run the frozen Parvec controlled-film source admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.parvec_controlled_film_source import run_source_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf3_a3n_parvec_controlled_film_source_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-request-order", action="store_true")
    args = parser.parse_args()
    order = (
        ("oembed", "watch", "project")
        if args.reverse_request_order
        else ("project", "watch", "oembed")
    )
    report = run_source_audit(args.config.resolve(), request_order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "observed_stock_ids": report["admission"]["observed_stock_ids"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
