#!/usr/bin/env python
"""Run the BO9 metadata-only B&W composite weak-pair source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_bw_composite_pair_source import audit_payload, canonical_bytes, fetch_search_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bo9_flickr_bw_composite_pair_source_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2bo9_flickr_bw_composite_pair_source_v1/report.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = audit_payload(fetch_search_payload(config, args.timeout_seconds), config)
    raw = canonical_bytes(report)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "branch": report["branch"],
                "complete_composites": report["metrics"]["complete_composites"],
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
