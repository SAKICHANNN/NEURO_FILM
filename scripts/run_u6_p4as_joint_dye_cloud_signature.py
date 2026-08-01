from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.joint_dye_cloud_signature import (
    evaluate_joint_dye_cloud_signature,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4as_joint_dye_cloud_mtf_nps_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4as_joint_dye_cloud_mtf_nps_v1/report.json",
    )
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate_joint_dye_cloud_signature(contract)
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={report['automatic_pass']}")


if __name__ == "__main__":
    main()
