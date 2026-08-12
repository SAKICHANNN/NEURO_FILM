#!/usr/bin/env python3
"""Run the frozen P4CW structured-scene scanner-flare experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.scanner_unmixing_layer_correlation import canonical_json
from src.eval.structured_scene_scanner_flare import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4cw_structured_scene_scanner_flare_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(ROOT, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["stable"]["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
