#!/usr/bin/env python3
"""Convert exact official-ROMM RGB16 TIFF to bounded Rec.2020 SDR RGB16 PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json
from src.preprocess import convert_official_romm_rgb16_to_rec2020_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.receipt is not None and args.receipt.exists():
        raise ValueError("receipt must be create-only")
    receipt = convert_official_romm_rgb16_to_rec2020_png(args.input, args.output)
    if args.receipt is not None:
        try:
            atomic_write_json(args.receipt, receipt)
        except Exception:
            args.output.unlink(missing_ok=True)
            raise
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
