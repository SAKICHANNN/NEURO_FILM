#!/usr/bin/env python3
"""Replay one validated recipe-history selection to a new exact output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_history_export import export_recipe_history_entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--recipe-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--maximum-recipe-files", type=int, default=10_000)
    parser.add_argument("--maximum-recipe-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--tile-size", type=int, default=None)
    args = parser.parse_args()
    if args.maximum_recipe_files < 1:
        parser.error("--maximum-recipe-files must be at least 1")
    if args.maximum_recipe_bytes < 1:
        parser.error("--maximum-recipe-bytes must be at least 1")
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be at least 1")
    receipt = export_recipe_history_entry(
        args.history_root,
        recipe_path=args.recipe_path,
        profile_path=args.profile,
        output_path=args.output,
        root=ROOT,
        maximum_recipe_files=args.maximum_recipe_files,
        maximum_recipe_bytes=args.maximum_recipe_bytes,
        tile_size=args.tile_size,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
