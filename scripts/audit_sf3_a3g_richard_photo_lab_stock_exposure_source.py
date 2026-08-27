#!/usr/bin/env python3
"""Run the frozen Richard Photo Lab stock-exposure source lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.richard_photo_lab_stock_source import run_source_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/sf3_a3g_richard_photo_lab_stock_exposure_source_lock_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_source_lock(args.config.resolve())
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
                "network_bytes_read": report["network_bytes_read"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
