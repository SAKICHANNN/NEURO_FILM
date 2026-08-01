#!/usr/bin/env python
"""Run the frozen BP3 B&W density-structure challenger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.flickr_bw_density_structure_challenger import evaluate, render_contact_sheet
from src.eval.flickr_single_author_pair_acquisition import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bp3_flickr_bw_density_structure_challenger_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2bp3_flickr_bw_density_structure_challenger_v1/report.json",
    )
    parser.add_argument("--contact-sheet", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(ROOT, config)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_sha = atomic_json(output, report)
    visual = None
    if args.contact_sheet is not None and report["automatic_pass"]:
        contact_path = args.contact_sheet if args.contact_sheet.is_absolute() else ROOT / args.contact_sheet
        visual = render_contact_sheet(ROOT, config, contact_path)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "branch": report["branch"],
                "metrics": report["metrics"],
                "report_sha256": report_sha,
                "stable_evidence_id": report["stable_evidence_id"],
                "visual": visual,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
