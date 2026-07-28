#!/usr/bin/env python
"""Build frozen AO6V three-way blind residual-comparison sheets."""

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


CONFIG_SHA256 = "762b4604e4ea6bb1507913a6aa1992268e0ab7f148467bfa38739b9f30e70119"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ao6v_b0_real_film_residual_visual_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2ao6v_b0_real_film_residual_visual_v1",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("AO6V config hash mismatch")
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
