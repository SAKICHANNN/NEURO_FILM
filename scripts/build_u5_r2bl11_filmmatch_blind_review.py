#!/usr/bin/env python3
"""Build deterministic BL11 versus AO6 blind review sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_luma_preserving_chroma_transplant import (  # noqa: E402
    build_blind_review_sheets,
)
from src.eval.filmmatch_strict_interior_fresh_confirmation import (  # noqa: E402
    validate_contract,
)


BL8_CONFIG = ROOT / "configs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validated = validate_contract(
        ROOT, json.loads(BL8_CONFIG.read_text(encoding="utf-8"))
    )
    candidate = (
        args.candidate_output_dir
        if args.candidate_output_dir.is_absolute()
        else ROOT / args.candidate_output_dir
    )
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    evidence = build_blind_review_sheets(
        root=ROOT,
        parent_output_dir=(
            ROOT
            / "outputs/u5_r2bl8_filmmatch_strict_interior_fresh_confirmation_v1/run_a"
        ),
        candidate_output_dir=candidate,
        source_rows=validated["source_rows"],
        source_ids=validated["eligible_ids"],
        output_dir=output,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
