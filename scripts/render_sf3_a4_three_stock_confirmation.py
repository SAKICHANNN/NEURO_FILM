from __future__ import annotations

import argparse
from pathlib import Path

from src.real_film.three_stock_confirmation_render import evaluate_and_materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--a2-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluate_and_materialize(
        args.contract,
        root=root,
        a2_report_path=args.a2_report,
        ledger_path=args.ledger,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
