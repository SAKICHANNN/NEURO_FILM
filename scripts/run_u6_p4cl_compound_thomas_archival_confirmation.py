"""Run the frozen U6.P4CL compound-Thomas archival confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.compound_thomas_archival_confirmation import (
    canonical_json,
    evaluate_compound_thomas,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p4cl_compound_thomas_archival_confirmation_v1.json",
    )
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_compound_thomas(
        ROOT, args.contract.resolve(), args.scratch.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(report["decision"])
    print(report["stable_evidence_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
