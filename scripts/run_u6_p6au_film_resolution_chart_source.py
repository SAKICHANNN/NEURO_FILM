"""Acquire the exact U6.P6AU Vision3 resolution-chart members."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.film_resolution_chart_source import acquire, load_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6au_film_resolution_chart_source_v1.json",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.config)
    try:
        report = acquire(contract, args.data_root)
    except ValueError as error:
        report = {
            "schema": "neuro-film.u6-p6au-film-resolution-chart-source-report.v1",
            "experiment_id": contract["experiment_id"],
            "automatic_pass": False,
            "failure_reason": str(error),
            "decision": contract["decision_if_fail"],
            "claim_ceiling": contract["claim_ceiling"],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
