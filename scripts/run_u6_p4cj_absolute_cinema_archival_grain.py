"""Run the frozen U6.P4CJ archival-film residual challenge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.absolute_cinema_archival_grain import (
    canonical_json,
    evaluate_absolute_cinema_archival_grain,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4cj_absolute_cinema_archival_grain_v1.json",
    )
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_absolute_cinema_archival_grain(
        ROOT, args.contract.resolve(), args.scratch.resolve()
    )
    encoded = canonical_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(report["decision"])
    print(report["stable_evidence_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
