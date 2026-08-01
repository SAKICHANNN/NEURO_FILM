#!/usr/bin/env python
"""Run U5.R2BO0's bounded metadata-only album audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_single_author_pair_source import audit_payload, fetch_album_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bo0_flickr_single_author_pair_source_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2bo0_flickr_single_author_pair_source_v1/report.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = audit_payload(fetch_album_payload(config, args.timeout_seconds), config)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    output.write_bytes(raw)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "branch": report["branch"],
                "eligible_pairs": report["metrics"]["eligible_complete_pairs"],
                "family_pair_counts": report["metrics"]["family_pair_counts"],
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
