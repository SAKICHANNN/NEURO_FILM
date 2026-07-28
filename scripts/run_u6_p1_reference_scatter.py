#!/usr/bin/env python
"""Run the frozen U6.P1A float64 reference-scatter witness."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_reference_scatter import (  # noqa: E402
    evaluate_reference_scatter,
    load_contract,
    write_report,
)

CONTRACT_SHA256 = "3f673a537205e95181a4342aec5bd77cedfbe43cf84ea83ea079d5623b31a22b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p1_reference_scatter_simulator_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical_contract_bytes = args.contract.read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    ).encode("utf-8")
    actual_contract_sha = hashlib.sha256(canonical_contract_bytes).hexdigest()
    if actual_contract_sha != CONTRACT_SHA256:
        raise ValueError(
            f"contract hash mismatch: {actual_contract_sha} != {CONTRACT_SHA256}"
        )
    report = evaluate_reference_scatter(load_contract(args.contract))
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
