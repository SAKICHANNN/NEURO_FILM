#!/usr/bin/env python3
"""Run the frozen bounded U6.P4CH official-Newson LOD evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.official_newson_gaussian_lod import (
    evaluate_official_newson_gaussian_lod,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4ch_official_newson_gaussian_lod_v1.json",
    )
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_lock, report = evaluate_official_newson_gaussian_lod(contract)
    source_sha = write_json(source_lock, args.source_lock)
    report_sha = write_json(report, args.output)
    print(
        json.dumps(
            {
                "source_lock_sha256": source_sha,
                "report_sha256": report_sha,
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
