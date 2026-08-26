from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.polut_three_stock_global_baseline import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen RF3.D11 PoLUT baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rf3_d11_polut_three_stock_global_baseline_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    args = parser.parse_args()
    report = evaluate(args.config, ROOT, args.output_dir, order=args.order)
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
