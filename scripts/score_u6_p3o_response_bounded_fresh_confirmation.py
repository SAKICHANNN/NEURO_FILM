#!/usr/bin/env python
"""Score the frozen U6.P3O blind confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.response_bounded_fresh_adjudication import (
    adjudicate_response_bounded_fresh_confirmation,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p3o_response_bounded_fresh_adjudication_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = adjudicate_response_bounded_fresh_confirmation(
        root=ROOT,
        contract=load_contract(args.config),
    )
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    report_sha256 = write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report_sha256,
                "stable_evidence_id": report["stable_evidence_id"],
                "decision": report["decision"],
                "passing_rounds": report["passing_rounds"],
                "confirmed_severe_artifact_count": report[
                    "confirmed_severe_artifact_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
