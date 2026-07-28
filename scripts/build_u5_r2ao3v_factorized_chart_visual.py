#!/usr/bin/env python
"""Build the frozen AO3V three-round blind visual sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.factorized_chart_visual import build_blind_sheets  # noqa: E402


CONFIG_SHA256 = "dd2c48998ac9ab1a67b4fbb47902fdd2804f057978198b29f571872aa80bbab7"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ao3v_factorized_chart_visual_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2ao3v_factorized_chart_visual_v1",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("AO3V config hash mismatch")
    result = build_blind_sheets(
        root=ROOT,
        config=json.loads(raw),
        output_dir=args.output_dir,
    )
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
