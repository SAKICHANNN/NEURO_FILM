#!/usr/bin/env python3
"""Run two fresh-process SF3.A0M capture-alignment stress replays."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_capture_alignment import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/sf3_a0m_three_stock_capture_alignment_stress_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/sf3_a0m_three_stock_capture_alignment_stress_v1")
    parser.add_argument("--worker-order", choices=("canonical", "reverse"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if (args.worker_order is None) != (args.worker_output is None):
        raise SystemExit("worker order and output must be supplied together")
    if args.worker_order:
        report = evaluate(load_contract(args.contract), ROOT, reverse=args.worker_order == "reverse")
        write_report(report, args.worker_output)
        return 0
    outputs = []
    for run_id, order in (("run_a", "canonical"), ("run_b", "reverse")):
        path = args.output_dir / run_id / "report.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--contract",
                str(args.contract),
                "--worker-order",
                order,
                "--worker-output",
                str(path),
            ],
            cwd=ROOT,
            check=True,
        )
        outputs.append(path.read_bytes())
    if outputs[0] != outputs[1]:
        raise RuntimeError("SF3.A0M fresh-process reports differ")
    report = json.loads(outputs[0])
    print(json.dumps({"report_sha256": __import__("hashlib").sha256(outputs[0]).hexdigest(), "report": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
