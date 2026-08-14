from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.gamutmlp_prophoto_source_audit import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u1_4c6_gamutmlp_nus_prophoto_source_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate(load_contract(args.contract), ROOT, args.output_dir)


if __name__ == "__main__":
    main()
