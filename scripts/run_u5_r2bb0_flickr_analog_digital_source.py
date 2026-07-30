#!/usr/bin/env python
"""Run the bounded Flickr Analog + Digital metadata-only source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_analog_digital_source import run_metadata_audit


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2bb0_flickr_analog_digital_source_v1/report.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = run_metadata_audit(
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(report)
    output.write_bytes(payload)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "visible_records": summary["visible_record_count"],
                "visible_owners": summary["visible_owner_count"],
                "derivative_rights_records": summary[
                    "derivative_rights_record_count"
                ],
                "explicit_pairs": summary[
                    "explicit_adjacent_candidate_pair_count"
                ],
                "rights_eligible_explicit_pairs": summary[
                    "rights_eligible_explicit_pair_count"
                ],
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
