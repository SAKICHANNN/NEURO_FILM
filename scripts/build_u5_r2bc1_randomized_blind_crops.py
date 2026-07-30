#!/usr/bin/env python3
"""Build per-row randomized BC1 crop sheets without revealing mappings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_density_grain_value import build_blind_crop_sheets
from src.eval.ao6_procedural_filmfx_value import sha256_file

CONFIG = ROOT / "configs" / "u5_r2bc1_ao6_density_grain_value_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    ids = list(config["population"]["expected_ids"])[:9]
    arms = list(config["arms"])
    generator = random.SystemRandom()
    sample_orders = []
    for _ in range(3):
        round_orders = {}
        for sample_id in ids:
            order = arms.copy()
            generator.shuffle(order)
            round_orders[sample_id] = order
        sample_orders.append(round_orders)
    blind = build_blind_crop_sheets(
        root=ROOT,
        config=config,
        report=report,
        output_dir=run_dir / "randomized_blind",
        sample_orders=sample_orders,
    )
    mapping_path = run_dir / "randomized_blind" / "mapping.json"
    mapping_path.write_text(
        json.dumps(blind, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sheet_hashes = {
        f"round_{index}": sha256_file(
            run_dir
            / "randomized_blind"
            / f"blind_round_{index}_crops.png"
        )
        for index in range(1, 4)
    }
    print(
        json.dumps(
            {
                "mapping_commitment_sha256": blind[
                    "mapping_commitment_sha256"
                ],
                "sheet_sha256": sheet_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
