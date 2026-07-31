"""Run the frozen U5.R2BK20 nonidentifiability witness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.unpaired_operator_nonidentifiability import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u5_r2bk20_unpaired_operator_nonidentifiability_v1.json",
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run(ROOT, ROOT / arguments.config, ROOT / arguments.output)
    print(report["decision"])
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
