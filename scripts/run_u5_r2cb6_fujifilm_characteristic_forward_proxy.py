#!/usr/bin/env python
"""Run U5.R2CB6 characteristic-constrained Fujifilm forward proxy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fujifilm_characteristic_forward_proxy import (
    evaluate_characteristic_forward_proxy,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2cb6_fujifilm_characteristic_forward_proxy_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_characteristic_forward_proxy(load_contract(args.contract), ROOT)
    digest = write_report(report, args.output)
    print(f"passed={report['passed']}")
    print(f"failed_gates={','.join(report['failed_gates'])}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
