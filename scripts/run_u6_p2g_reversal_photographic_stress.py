#!/usr/bin/env python
"""Run the frozen U6.P2G reversal photographic stress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.reversal_photographic_stress import (  # noqa: E402
    evaluate,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p2g_reversal_photographic_stress_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / contract["input"]["manifest"]).read_text(encoding="utf-8")
    )
    report = evaluate(
        contract,
        manifest,
        root=ROOT,
        contact_sheet_path=args.output_dir / "contact_sheet.png",
    )
    report_sha = write_report(report, args.output_dir / "report.json")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={report_sha}")
    print(f"contact_sheet_sha256={report['contact_sheet_sha256']}")


if __name__ == "__main__":
    main()
