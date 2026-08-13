from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.portra_scanner_nuisance_safe_residual_d0 import (
    evaluate,
    load_contract,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P6AK analytical safe residual diagnostic.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6ak_portra_scanner_nuisance_safe_residual_d0_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.config), ROOT)
    digest = write_report(report, args.output)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["decision"], "report_sha256": digest, "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
