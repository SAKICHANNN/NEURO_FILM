"""Run the frozen U6.P9E temporal exposure-flicker audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.temporal_exposure_flicker import load_contract, run_audit, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p9e_temporal_exposure_flicker_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(run_audit(root=ROOT, contract=load_contract(args.config)), args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
