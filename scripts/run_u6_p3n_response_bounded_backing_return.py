#!/usr/bin/env python
"""Run the U6.P3N response-bounded backing-return development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.response_bounded_backing_return import (
    evaluate_response_bounded_backing_return,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p3n_response_bounded_backing_return_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.config)
    report = evaluate_response_bounded_backing_return(
        root=ROOT,
        contract=contract,
        visual_root=args.visual_root,
    )
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report_sha256 = write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": report_sha256,
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_pass": report["automatic_pass"],
                "metrics": report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
