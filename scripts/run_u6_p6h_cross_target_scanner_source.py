#!/usr/bin/env python
"""Acquire and audit the frozen U6.P6H scanner source expansion."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_cross_target_scanner_source import (  # noqa: E402
    audit_cross_target_scanner_source,
    write_report,
)


CONTRACT_SHA256 = "7a567fd8ded4ca38b833f5eef4768e9d18be3451e0afc3c276f9a6149ab4a0d5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p6h_cross_target_scanner_source_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = args.contract.read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != CONTRACT_SHA256:
        raise ValueError(f"contract hash mismatch: {actual} != {CONTRACT_SHA256}")
    report = audit_cross_target_scanner_source(ROOT, args.contract)
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
