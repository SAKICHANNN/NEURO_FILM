#!/usr/bin/env python3
"""Run the frozen RF3.D5 spectral-shape discriminator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_spectral_shape import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rf3_d5_three_stock_spectral_shape_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.config)
    print(
        write_report(
            evaluate(contract, ROOT, overlay_dir=args.overlay_dir), args.output
        )
    )


if __name__ == "__main__":
    main()
