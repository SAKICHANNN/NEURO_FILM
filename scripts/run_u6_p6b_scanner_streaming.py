#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_scanner_streaming import (  # noqa: E402
    evaluate_scanner_streaming,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p6b_scanner_streaming_v1.json",
    )
    parser.add_argument(
        "--p6a-decision",
        type=Path,
        default=ROOT / "configs" / "u6_p6a_scanner_profile_decision_v1.json",
    )
    parser.add_argument(
        "--p6a-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p6a_scanner_profile_boundary_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    decision_raw = args.p6a_decision.read_bytes()
    if (
        hashlib.sha256(decision_raw).hexdigest()
        != contract["parents"]["p6a_decision_sha256"]
    ):
        raise ValueError("P6A decision hash does not match frozen parent")
    decision = json.loads(decision_raw.decode("utf-8"))
    p6a_raw = args.p6a_contract.read_bytes()
    if hashlib.sha256(p6a_raw).hexdigest() != decision["contract_sha256"]:
        raise ValueError("P6A contract hash does not match decision")
    report = evaluate_scanner_streaming(
        contract, json.loads(p6a_raw.decode("utf-8"))
    )
    report_sha = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")


if __name__ == "__main__":
    main()
