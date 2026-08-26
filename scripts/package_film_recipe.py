#!/usr/bin/env python3
"""Build or inspect deterministic no-pixel film-recipe recovery bundles."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    build_recipe_recovery_bundle,
    inspect_recipe_recovery_bundle,
    materialize_recipe_recovery_bundle,
    update_materialized_recipe_recovery_tree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or inspect a deterministic no-pixel recipe recovery bundle."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--recipe", type=Path, required=True)
    build.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    build.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--bundle", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    update = commands.add_parser("update")
    update.add_argument("--bundle", type=Path, required=True)
    update.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        result = build_recipe_recovery_bundle(
            recipe_path=args.recipe,
            profile_path=args.profile,
            root=ROOT,
            bundle_path=args.output,
        )
    elif args.command == "inspect":
        result = inspect_recipe_recovery_bundle(args.bundle)
    elif args.command == "restore":
        result = materialize_recipe_recovery_bundle(
            bundle_path=args.bundle,
            destination_root=args.destination,
        )
    else:
        result = update_materialized_recipe_recovery_tree(
            bundle_path=args.bundle,
            destination_root=args.destination,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
