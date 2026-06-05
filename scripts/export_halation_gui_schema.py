#!/usr/bin/env python3
"""Export the locked halation GUI schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import halation_gui_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export halation GUI schema JSON.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "schema" / "halation_gui_schema.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = halation_gui_schema()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
