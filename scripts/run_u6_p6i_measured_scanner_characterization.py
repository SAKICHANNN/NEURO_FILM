#!/usr/bin/env python
"""Run U6.P6I per-device measured-target scanner characterization."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_scanner_characterization import (  # noqa: E402
    evaluate_measured_scanner_characterization,
    write_report,
)


CONTRACT_SHA256 = "ae9d527e1008c78969e4f18b0629826e10be0f8e16e8137ffd7282794b1f425d"
ALIGNMENT_RETRY_SHA256 = (
    "c93c48047f6adaaaf6921fdac8b50c123e0705b97ecb6abc6f106c319ab9fb36"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p6i_measured_scanner_characterization_v1.json"
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
    retry_path = ROOT / "configs" / "u6_p6i1_alignment_retry_v1.json"
    retry_actual = hashlib.sha256(
        retry_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    if retry_actual != ALIGNMENT_RETRY_SHA256:
        raise ValueError(
            "alignment retry hash mismatch: "
            f"{retry_actual} != {ALIGNMENT_RETRY_SHA256}"
        )
    report = evaluate_measured_scanner_characterization(
        ROOT, args.contract, retry_path
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
