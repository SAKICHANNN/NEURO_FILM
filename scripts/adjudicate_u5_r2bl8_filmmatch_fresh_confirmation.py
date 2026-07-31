"""Adjudicate frozen BL8 preference plus BL9 non-basic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_fresh_confirmation_adjudication import (
    adjudicate_fresh_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads(
        (root / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1.json").read_text(encoding="utf-8")
    )
    bl8 = root / "outputs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1"
    bl9 = root / "outputs/u5_r2bl9_filmmatch_fresh_nonbasic_audit_v1"
    result = adjudicate_fresh_confirmation(
        config=config,
        observations_path=root / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_observations_v1.json",
        run_a=bl8 / "run_a",
        run_b=bl8 / "run_b",
        nonbasic_a_path=bl9 / "run_a.json",
        nonbasic_b_path=bl9 / "run_b.json",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "preference_gates": result["preference_gates"]}, sort_keys=True))
    print(result["stable_evidence_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
