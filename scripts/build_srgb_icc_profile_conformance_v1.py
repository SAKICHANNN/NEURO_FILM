"""Emit the exact portable sRGB ICC conformance artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.color_match.srgb_icc_profile import (
    srgb_icc_profile_conformance_v1,
)


def encode_conformance() -> str:
    return json.dumps(
        srgb_icc_profile_conformance_v1(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = encode_conformance()
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
