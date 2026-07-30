"""Run the frozen U6.P8BQ winner-only evaluator-Oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_fixed_arm_oracle import evaluate, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8bq_fixed_arm_oracle_diagnostic_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u6_p8bq_fixed_arm_oracle_diagnostic_v1/report.json"
        ),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(ROOT, config)
    write_report(args.output, report)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")


if __name__ == "__main__":
    main()
