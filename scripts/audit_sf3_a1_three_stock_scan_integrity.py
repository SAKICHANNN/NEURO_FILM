#!/usr/bin/env python3
"""Run the SF3.A1 controlled three-stock file/pixel integrity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_scan_integrity import (
    evaluate,
    evaluate_single_stock,
    materialize_alignment_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--build-alignment-evidence", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stock",
        choices=(
            "fujifilm_velvia_50",
            "kodak_portra_400",
            "kodak_ektar_100",
        ),
        help="Audit only one complete stock lane.",
    )
    args = parser.parse_args()
    if args.build_alignment_evidence:
        if args.manifest is not None:
            parser.error("--manifest is not used while building alignment evidence")
        report = materialize_alignment_evidence(args.contract, args.ledger, root=ROOT)
    else:
        if args.manifest is None:
            parser.error("--manifest is required for the A1 audit")
        report = (
            evaluate(args.contract, args.ledger, args.manifest, root=ROOT)
            if args.stock is None
            else evaluate_single_stock(
                args.contract,
                args.ledger,
                args.manifest,
                root=ROOT,
                stock=args.stock,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
