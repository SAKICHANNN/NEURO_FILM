"""Run U6.P2BB fresh photographic B&W structure value ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bw_structure_photographic_value import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p2bb_bw_structure_photographic_value_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = evaluate(
        root=ROOT,
        contract=load_contract(args.contract),
        output_dir=output_dir,
    )
    report_sha = write_report(report, output_dir / "report.json")
    print(f"report_sha256={report_sha}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={str(report['automatic_pass']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
