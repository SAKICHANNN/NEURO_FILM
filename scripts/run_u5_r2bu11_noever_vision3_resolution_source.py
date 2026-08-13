#!/usr/bin/env python3
"""Acquire or audit the frozen U5.R2BU11 source rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.noever_vision3_resolution_source import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2bu11_noever_vision3_resolution_source_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate(contract, ROOT, acquire=args.acquire)
    digest = write_report(report, args.output)
    print(json.dumps({"report_sha256": digest, **report}, sort_keys=True))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
