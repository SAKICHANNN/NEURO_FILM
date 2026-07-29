#!/usr/bin/env python
"""Run the frozen U6.P6E scanner RGB approximation audit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_spectral_scanner_rgb_approximation import (  # noqa: E402
    evaluate_spectral_scanner_rgb_approximation,
    load_contract,
    write_report,
)


CONTRACT_SHA256 = "f3da0aedf8f3836ef4d8fa5a0680bde7b03c1892c2d2d4af400b324ea8905878"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6e_spectral_scanner_rgb_approximation_v1.json"
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
    report = evaluate_spectral_scanner_rgb_approximation(
        ROOT, load_contract(args.contract)
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
