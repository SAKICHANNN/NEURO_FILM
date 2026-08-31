#!/usr/bin/env python3
"""Resume a hash-bound multi-input batch of the three product looks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.resumable_three_stock_input_batch import (
    render_resumable_three_stock_input_batch_to_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("workspace_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--look-amount", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-workers", type=int, default=1)
    parser.add_argument("--png-compression", type=int, choices=range(10), default=0)
    parser.add_argument("--maximum-new-jobs", type=int)
    args = parser.parse_args()
    result = render_resumable_three_stock_input_batch_to_directory(
        args.manifest,
        args.workspace_directory,
        args.output_directory,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        look_amount=args.look_amount,
        seed=args.seed,
        tile_size=args.tile_size,
        tile_workers=args.tile_workers,
        png_compression=args.png_compression,
        maximum_new_jobs=args.maximum_new_jobs,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
