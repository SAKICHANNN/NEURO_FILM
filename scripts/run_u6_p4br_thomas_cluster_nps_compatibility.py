from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.thomas_cluster_nps_compatibility import (
    evaluate_thomas_cluster_nps_compatibility,
    load_contract,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/u6_p4br_thomas_cluster_nps_compatibility_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/experiments/u6_p4br_thomas_cluster_nps_compatibility_v1/report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_thomas_cluster_nps_compatibility(
        load_contract(args.contract), ROOT
    )
    report_sha256 = write_report(report, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "output": str(args.output),
                "report_sha256": report_sha256,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
