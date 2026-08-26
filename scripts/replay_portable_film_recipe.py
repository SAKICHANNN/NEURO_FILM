#!/usr/bin/env python3
"""Explicitly bind and replay one private portable film recipe bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.portable_recipe_replay import (
    replay_portable_recipe_recovery_bundle_to_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bound-recipe", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--tile-size", type=int, default=None)
    args = parser.parse_args()
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be at least 1")
    receipt = replay_portable_recipe_recovery_bundle_to_file(
        bundle_path=args.bundle,
        input_path=args.input,
        output_path=args.output,
        recipe_path=args.bound_recipe,
        profile_path=args.profile,
        root=ROOT,
        tile_size=args.tile_size,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
