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
    build_ledger_template,
    build_receipt_template,
    build_single_stock_ledger_template,
    build_single_stock_receipt_template,
    evaluate_ledger_binding,
    evaluate_receipts,
    evaluate_single_stock_ledger_binding,
    evaluate_single_stock_receipts,
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
    parser.add_argument("--build-ledger-template", action="store_true")
    parser.add_argument(
        "--acquisition-contract",
        type=Path,
        default=ROOT / "configs/sf3_a0_three_stock_controlled_acquisition_v1.json",
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stock",
        choices=(
            "fujifilm_velvia_50",
            "kodak_portra_400",
            "kodak_ektar_100",
        ),
        help="Build or validate only one complete stock lane.",
    )
    args = parser.parse_args()
    if args.build_template:
        if (
            args.ledger is not None
            or args.manifest_output is not None
            or args.build_ledger_template
        ):
            parser.error("ledger outputs require --packet")
        report = (
            build_receipt_template(args.contract, root=ROOT)
            if args.stock is None
            else build_single_stock_receipt_template(
                args.contract, root=ROOT, stock=args.stock
            )
        )
    elif args.build_ledger_template:
        if args.ledger is not None or args.manifest_output is not None:
            parser.error("--build-ledger-template cannot compile a filled ledger")
        builder = (
            build_ledger_template
            if args.stock is None
            else build_single_stock_ledger_template
        )
        report = builder(
            args.contract,
            args.packet,
            args.acquisition_contract,
            root=ROOT,
            **({} if args.stock is None else {"stock": args.stock}),
        )
    elif args.ledger is not None:
        if args.manifest_output is None:
            parser.error("--ledger requires --manifest-output")
        binder = (
            evaluate_ledger_binding
            if args.stock is None
            else evaluate_single_stock_ledger_binding
        )
        report = binder(
            args.contract,
            args.packet,
            args.acquisition_contract,
            args.ledger,
            root=ROOT,
            **({} if args.stock is None else {"stock": args.stock}),
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
        report = (
            evaluate_receipts(args.contract, args.packet, root=ROOT)
            if args.stock is None
            else evaluate_single_stock_receipts(
                args.contract, args.packet, root=ROOT, stock=args.stock
            )
        )
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    if args.build_template or args.build_ledger_template:
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
