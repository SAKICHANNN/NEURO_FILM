#!/usr/bin/env python3
"""Build and optionally open the local offline film workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_workspace import (
    materialize_offline_recipe_workspace,
    open_offline_recipe_workspace,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--maximum-recipe-files", type=int, default=10_000)
    parser.add_argument("--maximum-recipe-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--open", action="store_true", dest="open_after_build")
    args = parser.parse_args()
    receipt = materialize_offline_recipe_workspace(
        args.history_root,
        args.workspace,
        maximum_recipe_files=args.maximum_recipe_files,
        maximum_recipe_bytes=args.maximum_recipe_bytes,
    )
    if args.open_after_build:
        open_offline_recipe_workspace(args.workspace)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
