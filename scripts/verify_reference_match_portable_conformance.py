#!/usr/bin/env python3
"""Verify the frozen language-neutral reference-match conformance bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.conformance import (  # noqa: E402
    load_portable_conformance_bundle,
    portable_conformance_result_to_json,
    verify_portable_conformance_bundle,
)
from src.inference import atomic_write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT
        / "configs"
        / "reference_match_portable_conformance_v1.json",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = verify_portable_conformance_bundle(
        load_portable_conformance_bundle(args.bundle)
    )
    encoded = portable_conformance_result_to_json(result)
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        atomic_write_json(args.output, json.loads(encoded))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
