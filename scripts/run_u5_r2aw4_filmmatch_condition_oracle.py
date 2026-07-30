"""Run the frozen FilmMatch controlled-condition operator Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_condition_oracle import (  # noqa: E402
    evaluate_condition_oracle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aw4_filmmatch_condition_oracle_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "condition_oracle_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_config = json.loads(
        (ROOT / config["parent"]["sample_config"]).read_text(encoding="utf-8")
    )
    parent_raw = (ROOT / config["parent"]["capacity_report"]).read_bytes()
    if (
        hashlib.sha256(parent_raw).hexdigest()
        != config["parent"]["capacity_report_sha256"]
        or json.loads(parent_raw)["stable_evidence_id"]
        != config["parent"]["capacity_stable_evidence_id"]
    ):
        raise ValueError("AW3 capacity parent identity drift")
    report = evaluate_condition_oracle(
        extract_pair_datasets(ROOT, parent_config), config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "oracle_gate_passed": report["oracle_gate_passed"],
                "conditional_router_research_opened": report[
                    "conditional_router_research_opened"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
