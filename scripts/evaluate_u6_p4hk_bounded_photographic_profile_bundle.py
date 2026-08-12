#!/usr/bin/env python3
"""Run the P4HK canonical profile-bundle audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bounded_photographic_profile_bundle import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = evaluate(contract, root=ROOT, bundle_path=args.bundle)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
