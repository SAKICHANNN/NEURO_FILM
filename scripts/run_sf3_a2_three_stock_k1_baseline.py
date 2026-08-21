#!/usr/bin/env python3
"""Run the file-backed SF3.A1 to SF3.A2 three-stock baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_k1_file_runner import evaluate_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--integrity-contract",
        type=Path,
        default=ROOT
        / "configs/sf3_a1_three_stock_file_pixel_alignment_integrity_v1.json",
    )
    parser.add_argument(
        "--k1-contract",
        type=Path,
        default=ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json",
    )
    args = parser.parse_args()
    report = evaluate_files(
        root=ROOT,
        integrity_contract_path=args.integrity_contract,
        k1_contract_path=args.k1_contract,
        ledger_path=args.ledger,
        manifest_path=args.manifest,
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
