"""Run P6AL analytical safe-residual scanner replication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.safe_residual_scanner_replication_d1 import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6al_safe_residual_scanner_replication_d1_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_contract(args.config), ROOT)
    digest = write_report(report, args.output)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["decision"], "report_sha256": digest, "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
