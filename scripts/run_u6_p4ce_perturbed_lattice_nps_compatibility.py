#!/usr/bin/env python3
"""Run the frozen U6.P4CE perturbed-lattice NPS compatibility test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.perturbed_lattice_nps_compatibility import (
    evaluate_perturbed_lattice_nps_compatibility,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4ce_perturbed_lattice_nps_compatibility_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate_perturbed_lattice_nps_compatibility(contract, ROOT)
    report_sha256 = write_report(report, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": report_sha256,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
