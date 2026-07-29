#!/usr/bin/env python
"""Run the frozen U6.P3D float64 backing-return reference."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_backing_return import (  # noqa: E402
    evaluate_backing_return,
    load_contract,
    load_legacy_control,
    write_report,
)


CONTRACT_SHA256 = "2b27173834345bf7d0492c213dea6b1d5ef8020d604046347f6915898ef6cc12"
LEGACY_CONTROL_SHA256 = (
    "3f673a537205e95181a4342aec5bd77cedfbe43cf84ea83ea079d5623b31a22b"
)


def _canonical_sha256(path: Path) -> str:
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    parser.add_argument(
        "--legacy-control",
        type=Path,
        default=Path("configs/u6_p1_reference_scatter_simulator_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_contract_sha = _canonical_sha256(args.contract)
    if actual_contract_sha != CONTRACT_SHA256:
        raise ValueError(
            f"contract hash mismatch: {actual_contract_sha} != {CONTRACT_SHA256}"
        )
    actual_legacy_sha = _canonical_sha256(args.legacy_control)
    if actual_legacy_sha != LEGACY_CONTROL_SHA256:
        raise ValueError(
            f"legacy control hash mismatch: {actual_legacy_sha} != "
            f"{LEGACY_CONTROL_SHA256}"
        )
    report = evaluate_backing_return(
        load_contract(args.contract),
        load_legacy_control(args.legacy_control),
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
