#!/usr/bin/env python3
"""Replay one exact supported-ProPhoto Rec.2020 render receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.prophoto_rec2020_replay import (
    replay_supported_prophoto_velvia_rec2020,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json",
    )
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    replay_supported_prophoto_velvia_rec2020(
        receipt,
        args.input,
        args.output,
        profile_path=args.profile,
        root=ROOT,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
