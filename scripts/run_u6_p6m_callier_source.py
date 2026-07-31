#!/usr/bin/env python
"""Acquire and audit the frozen U6.P6M primary Callier source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_callier_source import (  # noqa: E402
    acquire_source,
    audit_source,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6m_callier_source_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_contract(args.config)
    if not args.audit_only:
        acquire_source(config, ROOT)
    report = audit_source(config, ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Windows requires a writable descriptor for a durable file flush.
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
    return 0 if report["source_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
