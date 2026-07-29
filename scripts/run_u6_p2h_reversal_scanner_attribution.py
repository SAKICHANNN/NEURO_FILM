#!/usr/bin/env python
"""Run frozen U6.P2H scanner-stage attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.reversal_scanner_attribution import evaluate, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p2h_reversal_scanner_stage_attribution_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        json.loads(args.contract.read_text(encoding="utf-8")), root=ROOT
    )
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    print(
        "luma_owner="
        + report["attribution"]["largest_absolute_population_luma_shift"]
    )
    print(
        "spread_owner="
        + report["attribution"][
            "largest_absolute_population_channel_spread_shift"
        ]
    )


if __name__ == "__main__":
    main()
