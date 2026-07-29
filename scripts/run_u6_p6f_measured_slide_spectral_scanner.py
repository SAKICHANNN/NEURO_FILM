#!/usr/bin/env python
"""Run U6.P6F measured-slide spectra through the frozen synthetic scanners."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_slide_spectral_scanner import (  # noqa: E402
    evaluate_measured_slide_spectral_scanner,
    load_contract,
    write_report,
)


CONTRACT_SHA256 = "ebe3d441e338623c55f086850dc79d11b2a107a5bc8c773bf6b2f2c23ac11b41"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6f_measured_slide_spectral_scanner_v1.json"
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
    report = evaluate_measured_slide_spectral_scanner(
        ROOT, load_contract(args.contract)
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
