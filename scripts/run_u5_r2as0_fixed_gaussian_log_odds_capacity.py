#!/usr/bin/env python
"""Run the frozen U5.R2AS0 synthetic Gaussian-operator capacity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fixed_gaussian_log_odds_capacity import (  # noqa: E402
    evaluate_fixed_gaussian_log_odds_capacity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2as0_fixed_gaussian_log_odds_capacity_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_fixed_gaussian_log_odds_capacity(
        json.loads(args.config.read_text(encoding="utf-8")), ROOT
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "failed_checks": [
                    row["name"] for row in report["checks"] if not row["passed"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
