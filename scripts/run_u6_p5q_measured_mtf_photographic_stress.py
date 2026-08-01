#!/usr/bin/env python3
"""Run the frozen U6.P5Q photographic measured-MTF stress."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_mtf_photographic_stress import (
    evaluate_photographic_stress,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p5q_measured_mtf_photographic_stress_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overview", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate_photographic_stress(
        root=ROOT,
        contract=contract,
        overview_path=args.overview.resolve(),
        patch_path=args.patches.resolve(),
    )
    report["provenance"] = {
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
    }
    report_sha = write_report(report, args.report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": report_sha,
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
