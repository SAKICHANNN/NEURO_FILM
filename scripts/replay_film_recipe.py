#!/usr/bin/env python3
"""Regenerate an exact deterministic film render from one verified v1 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import replay_style_safe_recipe_to_file


def _replay_one(task: tuple[Path, Path, Path]) -> tuple[str, Path]:
    recipe_path, profile_path, output_path = task
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    digest = replay_style_safe_recipe_to_file(
        recipe,
        profile_path=profile_path,
        output_path=output_path,
        root=ROOT,
    )
    return digest, output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Replay independent recipes in up to this many worker processes.",
    )
    args = parser.parse_args()
    if len(args.recipe) != len(args.output):
        parser.error("--recipe and --output counts must match")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    tasks = [
        (recipe_path, args.profile, output_path)
        for recipe_path, output_path in zip(args.recipe, args.output, strict=True)
    ]
    if args.workers == 1:
        results = map(_replay_one, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)))
        results = executor.map(_replay_one, tasks)
    try:
        for digest, output_path in results:
            print(f"output_sha256={digest} output={output_path}")
    finally:
        if args.workers > 1:
            executor.shutdown(cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
