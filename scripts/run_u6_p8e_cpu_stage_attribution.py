#!/usr/bin/env python3
"""Run the U6.P8E canonical CPU stage attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_profile_stage_attribution import (  # noqa: E402
    evaluate_stage_attribution,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8e_cpu_stage_attribution_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_stage_attribution(root=ROOT, config=config)
    digest = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    for index, run in enumerate(report["runs"], start=1):
        print(f"run={index} total={run['total_attributed_seconds']:.3f}s")
        for name, row in run["stages"].items():
            print(
                f"  {name}: {row['elapsed_seconds']:.3f}s "
                f"rss={row['rss_bytes_after_stage'] / 2**30:.3f}GiB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
