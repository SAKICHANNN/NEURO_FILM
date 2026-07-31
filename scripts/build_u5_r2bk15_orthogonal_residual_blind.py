#!/usr/bin/env python3
"""Build frozen BK15 blind sheets without revealing mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.orthogonal_residual_blind_preference import (  # noqa: E402
    build_blind_sheets,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2bk15_orthogonal_residual_blind_preference_v1.json"
        ).read_text(encoding="utf-8")
    )
    receipt = build_blind_sheets(root=ROOT, config=config, output_dir=output)
    print(
        json.dumps(
            {
                "stable_evidence_id": receipt["stable_evidence_id"],
                "sheets": [
                    {
                        "round": row["round"],
                        "part": row["part"],
                        "sha256": row["sheet_sha256"],
                    }
                    for row in receipt["artifacts"]
                    if "sheet" in row
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
