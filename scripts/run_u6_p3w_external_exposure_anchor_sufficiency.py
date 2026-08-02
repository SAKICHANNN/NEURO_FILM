"""Run the frozen U6.P3W external exposure-anchor sufficiency ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.external_exposure_anchor_sufficiency import evaluate, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p3w_external_exposure_anchor_sufficiency_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(ROOT, args.contract)
    digest = write_report(report, args.output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "passed": report["passed"],
                "report_sha256": digest,
                "scientific_stable_id": report["scientific_stable_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
