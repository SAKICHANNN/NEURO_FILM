#!/usr/bin/env python3
"""Run U6.P8CR canonical Thomas export-profile evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_thomas_export_profile import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8cr_thomas_export_profile_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        root=ROOT,
        contract_path=args.contract.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(raw)
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_pass": report["automatic_pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
