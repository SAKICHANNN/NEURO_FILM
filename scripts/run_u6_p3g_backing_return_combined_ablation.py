#!/usr/bin/env python
"""Run the frozen U6.P3G backing-return topology ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_backing_return_combined_ablation import (  # noqa: E402
    evaluate_combined_ablation,
    load_contract,
    write_report,
)


def _exact_json(path: Path, expected_sha256: str, name: str):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} hash does not match frozen parent")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "u6_p3g_backing_return_combined_ablation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    parents = contract["parents"]
    manifest = _exact_json(
        ROOT / contract["input"]["manifest"],
        contract["input"]["manifest_sha256"],
        "manifest",
    )
    p1 = _exact_json(
        ROOT / parents["p1_contract_path"],
        parents["p1_contract_file_sha256"],
        "P1 contract",
    )
    p3d = json.loads((ROOT / parents["p3d_contract_path"]).read_text("utf-8"))
    sensitometry = _exact_json(
        ROOT / parents["sensitometry_path"],
        parents["sensitometry_file_sha256"],
        "sensitometry",
    )
    report = evaluate_combined_ablation(
        contract,
        manifest,
        p1,
        p3d,
        sensitometry,
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
