#!/usr/bin/env python3
"""Regenerate an exact deterministic film render from one verified v1 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import replay_style_safe_recipe_to_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--output", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.recipe) != len(args.output):
        parser.error("--recipe and --output counts must match")
    for recipe_path, output_path in zip(args.recipe, args.output, strict=True):
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        digest = replay_style_safe_recipe_to_file(
            recipe,
            profile_path=args.profile,
            output_path=output_path,
            root=ROOT,
        )
        print(f"output_sha256={digest} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
