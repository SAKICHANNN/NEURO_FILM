#!/usr/bin/env python
"""Run the frozen U6.P6X multispectral scanner compiler audit."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_multispectral_scanner_band_compiler import (
    evaluate_multispectral_scanner_band_compiler,
    load_contract,
    write_report,
)

CONTRACT_SHA256 = "72e8c41e824797ce79dc891240b202b69a7505434efff722128118753725c6ad"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6x_multispectral_scanner_band_compiler_v1.json"
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
    report = evaluate_multispectral_scanner_band_compiler(
        ROOT, load_contract(args.contract)
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"failed_gates={[name for name, value in report['checks'].items() if not value]}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
