#!/usr/bin/env python3
"""Adjudicate BL13 automatic and autonomous visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_nonbasic_chroma_adjudication import (  # noqa: E402
    adjudicate_ao6_nonbasic_chroma,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(
        (
            ROOT / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1.json"
        ).read_text(encoding="utf-8")
    )
    base = ROOT / "outputs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1"
    result = adjudicate_ao6_nonbasic_chroma(
        config=config,
        observations_path=(
            ROOT
            / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_observations_v1.json"
        ),
        run_a=base / "formal_a",
        run_b=base / "formal_b",
        blind_dir=base / "blind_v1",
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "candidate_choices_by_round": result[
                    "candidate_choices_by_round"
                ],
                "passing_rounds": result["passing_rounds"],
                "stable_evidence_id": result["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
