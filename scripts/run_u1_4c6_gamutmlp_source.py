from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.gamutmlp_prophoto_source_audit import evaluate, load_contract


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
