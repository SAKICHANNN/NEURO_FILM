#!/usr/bin/env python
"""Run the frozen U6.P5K measured-MTF discrete-kernel audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_mtf_kernel import evaluate_kernel, load_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p5k_measured_mtf_kernel_conformance_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_kernel(load_contract(args.config), ROOT, diagnostic_path=args.diagnostic)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, args.report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
