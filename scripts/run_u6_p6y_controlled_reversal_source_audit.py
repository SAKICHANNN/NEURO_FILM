#!/usr/bin/env python
"""Run the frozen U6.P6Y controlled reversal-film source audit."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.controlled_reversal_source_audit import (
    evaluate_controlled_reversal_source_audit,
    load_contract,
    write_report,
)

CONTRACT_SHA256 = "16aad43d91e8a6713e2a33d8d50e2cd6f55f40b0657c9620c18a89beba2fe9d5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p6y_controlled_reversal_source_audit_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = (
        args.contract.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    )
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != CONTRACT_SHA256:
        raise ValueError(f"contract hash mismatch: {actual} != {CONTRACT_SHA256}")
    report = evaluate_controlled_reversal_source_audit(
        ROOT, load_contract(args.contract)
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(
        "failed_gates="
        f"{[name for name, value in report['checks'].items() if not value]}"
    )
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
