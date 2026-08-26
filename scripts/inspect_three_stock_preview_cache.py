#!/usr/bin/env python3
"""Validate and return one existing three-stock preview cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview_cache import inspect_three_stock_preview_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview_directory", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/safe_rich_v1.json",
    )
    args = parser.parse_args()
    result = inspect_three_stock_preview_cache(
        args.preview_directory,
        input_path=args.input,
        profile_path=args.profile,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
