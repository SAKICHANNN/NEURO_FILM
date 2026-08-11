"""Run the frozen U6.P4CK AbsoluteDegradation release audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.absolute_degradation_blue_noise_release import (
    canonical_json,
    evaluate_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p4ck_absolute_degradation_blue_noise_release_audit_v1.json",
    )
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_release(args.contract.resolve(), args.scratch.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report))
    print(report["decision"])
    print(report["stable_evidence_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
