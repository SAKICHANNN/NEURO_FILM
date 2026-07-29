#!/usr/bin/env python
"""Run the fixed U5.R2AR1 group-balanced display-proxy audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.velvia_group_balanced_operator import (  # noqa: E402
    evaluate_velvia_group_balanced_operator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/u5_r2ar1_velvia_group_balanced_operator_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_velvia_group_balanced_operator(
        json.loads(args.config.read_bytes()), ROOT
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                **report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
