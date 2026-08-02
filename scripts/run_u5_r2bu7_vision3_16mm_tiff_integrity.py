#!/usr/bin/env python
"""Run the frozen U5.R2BU7 TIFF integrity and role audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.vision3_16mm_tiff_integrity import evaluate_tiff_integrity, load_contract


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bu7_vision3_16mm_tiff_integrity_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_tiff_integrity(ROOT, load_contract(args.config))
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "gate_pass": report["gate_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "downloaded_file_count": report["downloaded_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
