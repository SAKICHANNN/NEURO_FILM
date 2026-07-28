#!/usr/bin/env python3
"""Run the frozen U6.P8C ingress and receipt audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_profile_ingress import (  # noqa: E402
    evaluate_profile_ingress,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8c_profile_ingress_receipt_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_profile_ingress(root=ROOT, config=config)
    digest = write_report(report, args.output)
    print(f"decision={report['decision']}")
    print(f"artifact_bytes={report['artifact_bytes']}")
    print(f"artifact_bytes_sha256={report['artifact_bytes_sha256']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
