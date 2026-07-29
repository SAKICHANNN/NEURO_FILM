#!/usr/bin/env python3
"""Adjudicate the frozen U6.P8BN blind strength comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fresh_strength_adjudication import (  # noqa: E402
    adjudicate_fresh_strength,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u6_p8bn_fresh_strength_adjudication_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "u6_p8bn_fresh_strength_population_v1"
        / "adjudication.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = adjudicate_fresh_strength(root=ROOT, config=config)
    digest = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(f"strength_0.8_round_wins={report['strength_round_wins']['0.8']}")
    print(f"strength_1.0_round_wins={report['strength_round_wins']['1.0']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
