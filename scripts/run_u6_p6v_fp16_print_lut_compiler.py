"""Run the frozen U6.P6V FP16 print-LUT compiler audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fp16_print_lut_compiler import load_contract, run_audit, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6v_fp16_print_lut_compiler_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_audit(root=ROOT, contract=load_contract(arguments.config))
    print(write_report(report, arguments.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
