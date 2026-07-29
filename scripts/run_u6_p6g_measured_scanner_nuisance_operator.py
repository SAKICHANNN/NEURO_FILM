#!/usr/bin/env python
"""Run U6.P6G on the frozen same-slide real-scanner evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_scanner_nuisance_operator import (  # noqa: E402
    evaluate_measured_scanner_nuisance_operator,
    write_report,
)


CONTRACT_SHA256 = "a2d8ddb9a60c317738e3e0d800e0457ea9e0dd0f8213b45963f6950e06f8b9d3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6g_measured_scanner_nuisance_operator_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = args.contract.read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != CONTRACT_SHA256:
        raise ValueError(f"contract hash mismatch: {actual} != {CONTRACT_SHA256}")
    report = evaluate_measured_scanner_nuisance_operator(
        ROOT, args.contract
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
