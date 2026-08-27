#!/usr/bin/env python3
"""Run SF3.A2B on an already sampled three-stock NPZ packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_k1_baseline import StockFrameSamples
from src.real_film.three_stock_pooled_global_control import evaluate


def _load_packet(
    path: Path,
) -> tuple[dict[str, list[StockFrameSamples]], dict[str, list[StockFrameSamples]]]:
    development: dict[str, list[StockFrameSamples]] = {}
    confirmation: dict[str, list[StockFrameSamples]] = {}
    with np.load(path, allow_pickle=False) as packet:
        manifest = json.loads(str(packet["manifest"].item()))
        for role, destination in (
            ("development", development),
            ("confirmation", confirmation),
        ):
            for stock, rows in manifest[role].items():
                destination[stock] = [
                    StockFrameSamples(
                        scene_id=row["scene_id"],
                        frame_id=row["frame_id"],
                        roll_id=row["roll_id"],
                        source=packet[row["source_key"]],
                        target=packet[row["target_key"]],
                    )
                    for row in rows
                ]
    return development, confirmation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a2b_three_stock_pooled_global_control_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    development, confirmation = _load_packet(args.packet)
    report = evaluate(
        args.contract, root=ROOT, development=development, confirmation=confirmation
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
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
