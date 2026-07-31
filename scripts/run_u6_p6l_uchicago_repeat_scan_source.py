#!/usr/bin/env python
"""Acquire and audit the frozen U6.P6L same-plate repeat-scan subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_repeat_scan_source import (  # noqa: E402
    acquire_frozen_subset,
    audit_repeat_scan_source,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6l_uchicago_repeat_scan_source_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_contract(args.config)
    if not args.audit_only:
        acquire_frozen_subset(config, ROOT)
    report = audit_repeat_scan_source(config, ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
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
