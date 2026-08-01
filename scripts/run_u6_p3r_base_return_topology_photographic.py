#!/usr/bin/env python
"""Run the frozen U6.P3R photographic topology comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.base_return_topology_photographic import (
    evaluate_photographic_topology,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u6_p3r_base_return_topology_photographic_v1.json")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_photographic_topology(root=ROOT, config=load_contract(args.config), visual_root=args.visual_root)
    report_sha = write_report(report, args.report)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "report_sha256": report_sha, "stable_evidence_id": report["stable_evidence_id"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
