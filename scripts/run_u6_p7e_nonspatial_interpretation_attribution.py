#!/usr/bin/env python3
"""Run U6.P7E non-spatial interpretation attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_nonspatial_attribution import (  # noqa: E402
    evaluate_attribution,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u6_p7e_nonspatial_interpretation_attribution_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_attribution(
        root=ROOT,
        config=config,
        output_dir=args.output_dir,
    )
    digest = write_report(report, args.output_dir / "report.json")
    print(f"selected_candidate_arm_id={report['selected_candidate_arm_id']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
