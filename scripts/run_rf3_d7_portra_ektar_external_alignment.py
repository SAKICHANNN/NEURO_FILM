#!/usr/bin/env python3
"""Run the frozen RF3.D7 Portra/Ektar external-alignment control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.portra_ektar_external_alignment import evaluate
from src.inference import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/rf3_d7_portra_ektar_external_alignment_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = evaluate(root=ROOT, contract_path=args.contract, reverse=args.reverse)
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
