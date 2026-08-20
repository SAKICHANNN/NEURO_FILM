#!/usr/bin/env python3
"""Run the frozen CanonCGT extended-domain quadratic inverse preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.canoncgt_extended_quadratic_inverse import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid8_canoncgt_extended_quadratic_inverse_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = run_preflight(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        output_path=args.output.resolve(),
        device=args.device,
        reverse=args.reverse,
    )
    print(json.dumps({"path": str(result["path"]), "sha256": result["sha256"], "decision": result["report"]["decision"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
