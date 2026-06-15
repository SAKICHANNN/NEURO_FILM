#!/usr/bin/env python3
"""Inspect input image metadata for the shared preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import inspect_input, load_working_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an input image for preprocessing metadata.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--load", action="store_true", help="Also decode to WorkingImage and report pixel stats.")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_input(args.input).to_dict()
    if args.load:
        working = load_working_image(args.input)
        report["working_image"] = {
            "shape": list(working.pixels.shape),
            "dtype": str(working.pixels.dtype),
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "min": float(working.pixels.min()),
            "max": float(working.pixels.max()),
            "mean": float(working.pixels.mean()),
            "alpha_policy": working.alpha_policy,
            "warnings": [warning.__dict__ for warning in working.warnings],
        }
    text = json.dumps(report, indent=2 if args.pretty else None)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
