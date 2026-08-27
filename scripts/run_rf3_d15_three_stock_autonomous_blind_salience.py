#!/usr/bin/env python3
"""Build or adjudicate the frozen RF3.D15 blind salience package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.three_stock_autonomous_blind_salience import (
    adjudicate_package,
    build_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "adjudicate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rf3_d15_three_stock_autonomous_blind_salience_v1.json",
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--observations-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        report = build_package(args.config, ROOT, args.package_dir)
    else:
        if (
            args.observations is None
            or not args.observations_commit
            or args.output is None
        ):
            parser.error(
                "adjudicate requires --observations, --observations-commit and --output"
            )
        report = adjudicate_package(
            args.config,
            ROOT,
            args.package_dir,
            args.observations,
            args.observations_commit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
