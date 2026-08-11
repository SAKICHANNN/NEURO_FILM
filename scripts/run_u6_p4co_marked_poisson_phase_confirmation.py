from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.marked_poisson_phase_confirmation import (
    canonical_json,
    evaluate_marked_poisson_phase,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4co_marked_poisson_phase_confirmation_v1.json",
    )
    parser.add_argument(
        "--scratch", type=Path, default=Path("D:/trace-019f4b76-p4co-absolute-cinema")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_marked_poisson_phase(ROOT, args.contract, args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
