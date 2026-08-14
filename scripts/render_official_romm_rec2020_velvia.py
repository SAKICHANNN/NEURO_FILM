#!/usr/bin/env python3
"""Render strict official-ROMM RGB16 through the Rec.2020 Velvia look."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia import render_official_romm_velvia_rec2020


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/render_profiles/romm_rec2020_velvia_v1.json",
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt is not None and args.receipt.exists():
        raise ValueError("receipt must be create-only")
    receipt = render_official_romm_velvia_rec2020(
        args.input, args.output, profile_path=args.profile, root=ROOT
    )
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
