#!/usr/bin/env python
"""Run the frozen U6.P7A1 colour/developed-spatial ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_joint_ablation import (  # noqa: E402
    evaluate_ablation,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "u6_p7a1_interpretation_bounded_ablation_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    correction = json.loads(args.contract.read_text(encoding="utf-8"))
    report = evaluate_ablation(
        root=ROOT,
        correction=correction,
        output_dir=args.output_dir,
    )
    digest = write_report(report, args.output_dir / "report.json")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
