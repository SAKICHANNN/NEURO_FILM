#!/usr/bin/env python3
"""Build or execute portable offline recipe export requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_export_request import (
    export_recipe_request,
    materialize_recipe_export_request_set,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-request-set", type=Path)
    action.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    parser.add_argument("--maximum-recipe-files", type=int, default=10_000)
    parser.add_argument("--maximum-recipe-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--maximum-request-bytes", type=int, default=64 * 1024)
    parser.add_argument("--tile-size", type=int)
    args = parser.parse_args()
    if args.build_request_set is not None:
        if args.output is not None:
            parser.error("--output is only valid with --request")
        receipt = materialize_recipe_export_request_set(
            args.history_root,
            args.build_request_set,
            maximum_recipe_files=args.maximum_recipe_files,
            maximum_recipe_bytes=args.maximum_recipe_bytes,
        )
    else:
        if args.output is None:
            parser.error("--output is required with --request")
        receipt = export_recipe_request(
            args.history_root,
            args.request,
            profile_path=args.profile,
            output_path=args.output,
            root=ROOT,
            maximum_recipe_files=args.maximum_recipe_files,
            maximum_recipe_bytes=args.maximum_recipe_bytes,
            maximum_request_bytes=args.maximum_request_bytes,
            tile_size=args.tile_size,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
