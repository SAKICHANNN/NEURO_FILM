#!/usr/bin/env python
"""Build the frozen AY7 blinded product-value packs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_monotone_curve_blind_review import (  # noqa: E402
    build_packs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay7_monotone_curve_blind_product_value_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-side", type=int, default=1024)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build_packs(
        root=ROOT,
        config=config,
        output_dir=args.output_dir,
        maximum_side=args.maximum_side,
    )
    print(
        json.dumps(
            {
                "pack_sha256": result["pack_sha256"],
                "rounds": len(result["pack"]["rounds"]),
                "selected_pair_count": result["pack"][
                    "selected_pair_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
