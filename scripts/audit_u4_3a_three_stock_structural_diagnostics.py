from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.three_stock_structural_diagnostics import run_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_3a_three_stock_structural_diagnostics_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_diagnostics(config, ROOT, reverse=args.reverse)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
