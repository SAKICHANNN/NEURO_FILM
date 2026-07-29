#!/usr/bin/env python
"""Build AP3V blind sheets for both frozen Ektachrome shortlist rows."""

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


CONFIG_SHA256 = "20f4a51849338b925b0676cf3b6863a730e3dc31d007c6a7d7f888575388311c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ap3v_b0_ektachrome_residual_visual_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2ap3v_b0_ektachrome_residual_visual_v1",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("AP3V config hash mismatch")
    config = json.loads(raw)
    report_path = ROOT / str(config["parent_automatic_report"])
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != config[
        "parent_automatic_report_sha256"
    ]:
        raise ValueError("AP3V parent report hash mismatch")

    results = {}
    for candidate in config["shortlist"]:
        comparison = {
            **config,
            "experiment_id": candidate["experiment_id"],
            "looks": [*config["comparators"], candidate],
        }
        results[candidate["candidate_id"]] = build_three_way_blind_sheets(
            root=ROOT,
            config=comparison,
            output_dir=args.output_dir / candidate["candidate_id"],
        )
    output = {
        "config_sha256": CONFIG_SHA256,
        "comparisons": results,
        "claim_ceiling": config["claim_ceiling"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "build_report.json"
    report.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
