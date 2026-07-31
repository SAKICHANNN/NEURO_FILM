"""Adjudicate the frozen U5.R2BL6 blind comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_strict_interior_ood_adjudication import (
    adjudicate_strict_interior_ood,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads(
        (root / "configs/u5_r2bl6_filmmatch_strict_interior_ood_v1.json")
        .read_text(encoding="utf-8")
    )
    result = adjudicate_strict_interior_ood(
        config=config,
        observations_path=root
        / "configs/u5_r2bl6_filmmatch_strict_interior_ood_observations_v1.json",
        run_a=(root / args.run_a).resolve(),
        run_b=(root / args.run_b).resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gates"], sort_keys=True))
    print(result["stable_evidence_id"])
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
