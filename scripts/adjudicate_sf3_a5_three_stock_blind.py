"""Adjudicate frozen SF3.A5 observations after mapping reveal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_blind_package import adjudicate_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a5_three_stock_blind_distinguishability_v1.json",
    )
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--public-sheet", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = adjudicate_package(
        args.contract,
        package_report_path=args.package_report,
        public_sheet_path=args.public_sheet,
        private_mapping_path=args.private_mapping,
        observations_path=args.observations,
        reveal_path=args.reveal,
    )
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("automatic_pass", "decision", "stable_evidence_id")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
