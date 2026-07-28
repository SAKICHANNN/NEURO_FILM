#!/usr/bin/env python
"""Run U6.P7C cumulative spatial-stage attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_spatial_stage_attribution import (  # noqa: E402
    evaluate_stage_attribution,
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
            / "u6_p7c_spatial_stage_attribution_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.contract.read_text(encoding="utf-8"))
    report = evaluate_stage_attribution(
        root=ROOT, config=config, output_dir=args.output_dir
    )
    digest = write_report(report, args.output_dir / "report.json")
    print(f"dominant_stage={report['summary']['dominant_edge_loss_stage']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
