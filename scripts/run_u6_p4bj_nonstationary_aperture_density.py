from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.nonstationary_aperture_density import (
    evaluate_nonstationary_aperture_density,
    load_contract,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/u6_p4bj_nonstationary_aperture_density_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/experiments/u6_p4bj_nonstationary_aperture_density_v1/report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_nonstationary_aperture_density(load_contract(args.contract), ROOT)
    print(f"report_sha256={write_report(report, args.output)}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"decision={report['decision']}")


if __name__ == "__main__":
    main()
