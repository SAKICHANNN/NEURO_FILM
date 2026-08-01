#!/usr/bin/env python
"""Audit the frozen U6.P4AT NASA numerical MTF/granularity tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.nasa_joint_mtf_granularity_source import (
    audit_source,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4at_nasa_joint_mtf_granularity_source_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit_source(load_contract(args.config), ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(
        json.dumps(
            {
                "source_pass": report["source_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
