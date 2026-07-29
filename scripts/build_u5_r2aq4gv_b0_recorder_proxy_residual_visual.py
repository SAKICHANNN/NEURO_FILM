#!/usr/bin/env python
"""Build frozen AQ4GV blind recorder-residual comparison sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.three_way_look_visual import (  # noqa: E402
    build_three_way_blind_sheets,
)


CONFIG_SHA256 = "e192a32d987ee08d91c1dba3e32fc95d543b31a19f5b09a5a8bad5c3e3233ad5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2aq4gv_b0_recorder_proxy_residual_visual_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2aq4gv_b0_recorder_proxy_residual_visual_v1"
        ),
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("AQ4GV config hash mismatch")
    result = build_three_way_blind_sheets(
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
