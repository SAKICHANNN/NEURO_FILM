"""Run the frozen U6.P2AD first-party source audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.trix_developer_contrast_source import (
    load_contract,
    run_audit,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u6_p2ad_trix_developer_contrast_source_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(root=ROOT, contract=load_contract(args.contract))
    digest = write_report(report, args.output)
    print(f"report_sha256={digest}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"automatic_pass={str(report['automatic_pass']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
