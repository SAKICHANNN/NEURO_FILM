#!/usr/bin/env python
"""Compile the frozen U6.P4AX amplitude-only profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_250d_granularity_amplitude_profile import (
    compile_and_evaluate,
    load_contract,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p4ax_kodak_250d_granularity_amplitude_profile_v1.json",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle, report = compile_and_evaluate(load_contract(args.contract), ROOT)
    bundle_sha = write_json(bundle, args.bundle)
    report_sha = write_json(report, args.report)
    print(f"bundle_sha256={bundle_sha}")
    print(f"profile_id={report['profile_id']}")
    print(f"report_sha256={report_sha}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={report['automatic_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
