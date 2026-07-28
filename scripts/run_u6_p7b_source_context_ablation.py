#!/usr/bin/env python
"""Run U6.P7B source-context-fixed joint ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_joint_source_context import (  # noqa: E402
    evaluate_source_context_ablation,
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
            / "u6_p7b_source_context_joint_ablation_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.contract.read_text(encoding="utf-8"))
    report = evaluate_source_context_ablation(
        root=ROOT, config=config, output_dir=args.output_dir
    )
    digest = write_report(report, args.output_dir / "report.json")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
