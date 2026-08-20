#!/usr/bin/env python3
"""Execute the frozen RGB2RAW capture-pair pixel preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.rgb2raw_capture_pair_pixel_preflight import run_pixel_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf3_a0s_rgb2raw_capture_pair_pixel_preflight_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_pixel_preflight(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_identity": report["stable_identity"],
            }
        )
    )


if __name__ == "__main__":
    main()
