"""Compile real SF3 acquisition files into a hash-only manifest and audit it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_acquisition import (
    ThreeStockAcquisitionError,
    compile_acquisition_ledger,
    evaluate_manifest,
    evaluate_single_stock_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "sf3_a0_three_stock_controlled_acquisition_v1.json",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--stock",
        choices=(
            "fujifilm_velvia_50",
            "kodak_portra_400",
            "kodak_ektar_100",
        ),
        help="Admit only one complete stock lane.",
    )
    args = parser.parse_args()
    manifest = compile_acquisition_ledger(args.contract, args.ledger, root=ROOT)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output_manifest.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise ThreeStockAcquisitionError(
            f"refusing to overwrite manifest: {args.output_manifest}"
        ) from exc
    report = (
        evaluate_manifest(args.contract, args.output_manifest)
        if args.stock is None
        else evaluate_single_stock_manifest(
            args.contract, args.output_manifest, stock=args.stock
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ThreeStockAcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
