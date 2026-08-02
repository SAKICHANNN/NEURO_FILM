#!/usr/bin/env python
"""Run the frozen U5.R2BU6 contact-sheet source audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.vision3_16mm_contact_sheet_source import (
    evaluate_contact_sheet_source,
    load_contract,
    load_observations,
)


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
        default=ROOT / "configs/u5_r2bu6_vision3_16mm_contact_sheet_source_v1.json",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT
        / "configs/u5_r2bu6_vision3_16mm_contact_sheet_observations_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_contact_sheet_source(
        ROOT,
        load_contract(args.config),
        load_observations(args.observations),
    )
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "gate_pass": report["gate_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "embedded_image_count": report["pdf_facts"][
                    "embedded_image_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
