#!/usr/bin/env python3
"""Reveal and adjudicate frozen BK9 observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.smooth_perceptual_blind_preference import (  # noqa: E402
    adjudicate_blind,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_dir = (
        args.build_dir
        if args.build_dir.is_absolute()
        else ROOT / args.build_dir
    )
    observations = (
        args.observations
        if args.observations.is_absolute()
        else ROOT / args.observations
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError("BK9 result is create-only")
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2bk9_smooth_perceptual_blind_preference_v1.json"
        ).read_text(encoding="utf-8")
    )
    result = adjudicate_blind(
        root=ROOT,
        config=config,
        build_dir=build_dir,
        observations_path=observations,
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
