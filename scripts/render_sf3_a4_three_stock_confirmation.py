from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.three_stock_confirmation_render import (
    evaluate_and_materialize,
    evaluate_single_stock_and_materialize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--a2-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stock",
        choices=(
            "fujifilm_velvia_50",
            "kodak_portra_400",
            "kodak_ektar_100",
        ),
        help="Render one passing stock lane without cross-stock claims.",
    )
    args = parser.parse_args()
    evaluator = (
        evaluate_and_materialize
        if args.stock is None
        else evaluate_single_stock_and_materialize
    )
    kwargs = {
        "root": ROOT,
        "a2_report_path": args.a2_report,
        "ledger_path": args.ledger,
        "manifest_path": args.manifest,
        "output_dir": args.output_dir,
    }
    if args.stock is not None:
        kwargs["stock"] = args.stock
    evaluator(
        args.contract,
        **kwargs,
    )


if __name__ == "__main__":
    main()
