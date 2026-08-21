"""Audit one filled SF3.A0N physical capture receipt packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_capture_receipts import (
    build_receipt_template,
    evaluate_ledger_binding,
    evaluate_receipts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a0n_three_stock_capture_receipts_v1.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--packet", type=Path)
    mode.add_argument("--build-template", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--acquisition-contract",
        type=Path,
        default=ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json",
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.build_template:
        if args.ledger is not None or args.manifest_output is not None:
            parser.error("ledger outputs require --packet")
        report = build_receipt_template(args.contract, root=ROOT)
    elif args.ledger is not None:
        if args.manifest_output is None:
            parser.error("--ledger requires --manifest-output")
        report = evaluate_ledger_binding(
            args.contract,
            args.packet,
            args.acquisition_contract,
            args.ledger,
            root=ROOT,
        )
        manifest = report.pop("compiled_manifest")
        manifest_payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_output.open("xb") as handle:
            handle.write(manifest_payload)
    else:
        if args.manifest_output is not None:
            parser.error("--manifest-output requires --ledger")
        report = evaluate_receipts(args.contract, args.packet, root=ROOT)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    if args.build_template:
        print(json.dumps({"output": str(args.output), "template_only": True}))
    else:
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("automatic_pass", "decision", "stable_evidence_id")
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
