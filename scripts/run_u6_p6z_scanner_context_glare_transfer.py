#!/usr/bin/env python
"""Run the frozen U6.P6Z measured scanner context-glare transfer audit."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scanner_context_glare_transfer import (
    evaluate_scanner_context_glare_transfer,
    load_contract,
    load_source_table,
    write_report,
)

CONTRACT_SHA256 = "0eb6bcbedc9a56205d6fc9b0788c44e899020a0d93a7d57f6332ea0b3b1ead10"


def _canonical_file_sha256(path: Path) -> str:
    payload = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p6z_scanner_context_glare_transfer_v1.json"),
    )
    parser.add_argument(
        "--source-table",
        type=Path,
        default=Path("configs/u6_p6z_scanner_context_glare_table_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual = _canonical_file_sha256(args.contract)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"contract hash mismatch: {actual} != {CONTRACT_SHA256}")
    report = evaluate_scanner_context_glare_transfer(
        load_contract(args.contract), load_source_table(ROOT, args.source_table)
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"failed_gates={report['failed_gates']}")
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
