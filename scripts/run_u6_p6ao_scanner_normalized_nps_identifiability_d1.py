from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scanner_normalized_nps_identifiability_d1 import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = load_contract(ROOT / args.config)
    report = evaluate(contract, ROOT)
    digest = write_report(report, ROOT / args.output)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["decision"], "report_sha256": digest, "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
