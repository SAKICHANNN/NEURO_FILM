from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.gamutmlp_rec2020_stress_confirmation import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u1_4c7_gamutmlp_rec2020_stress_confirmation_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.contract), ROOT, args.output_dir)
    report_sha256 = write_report(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "report_sha256": report_sha256,
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
