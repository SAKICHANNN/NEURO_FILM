#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.nonnegative_y_eigen_matrix_transport import evaluate, load_contract
from src.eval.safe_base_ao6_chroma_direction import write_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u5_r2cb49_nonnegative_y_eigen_matrix_development_v1.json",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    report = evaluate(load_contract(a.contract), ROOT, a.output_dir)
    digest = write_report(report, a.report)
    print(
        f"automatic_pass={report['automatic_pass']}\ndecision={report['decision']}\nstable_evidence_id={report['stable_evidence_id']}\nreport_sha256={digest}"
    )


if __name__ == "__main__":
    main()
