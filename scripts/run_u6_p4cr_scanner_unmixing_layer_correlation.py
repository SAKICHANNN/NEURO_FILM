#!/usr/bin/env python3
"""Run the frozen P4CR scanner-unmixing correlation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scanner_unmixing_layer_correlation import canonical_json, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u6_p4cr_scanner_unmixing_layer_correlation_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(ROOT, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["stable"]["decision"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
