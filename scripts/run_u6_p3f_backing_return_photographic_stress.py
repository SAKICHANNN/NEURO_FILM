#!/usr/bin/env python
"""Run the frozen U6.P3F backing-return photographic stress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_backing_return_photographic_stress import (  # noqa: E402
    evaluate_photographic_stress,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p3f_backing_return_photographic_stress_v1.json"),
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    manifest_path = ROOT / contract["input"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    report = evaluate_photographic_stress(
        contract,
        manifest,
        parent,
        root=ROOT,
        contact_sheet_path=args.contact_sheet,
    )
    digest = write_report(report, args.output)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"contact_sheet_sha256={report['contact_sheet_sha256']}")
    print(f"report_sha256={digest}")


if __name__ == "__main__":
    main()
