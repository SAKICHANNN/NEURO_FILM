#!/usr/bin/env python
"""Build the frozen U5.R2AY0S FiveK paired source evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_neutral_base_source import build_source_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay0_fivek_neutral_base_source_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/u5_r2ay0_fivek_neutral_base_source_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = build_source_evidence(
        root=ROOT,
        config=config,
        config_path=config_path,
        output_dir=args.output.resolve(),
        software_commit=commit,
    )
    print(
        json.dumps(
            {
                "automatic_pass": result["report"]["automatic_pass"],
                "pair_count": result["report"]["pair_count"],
                "camera_model_group_count": result["report"][
                    "camera_model_group_count"
                ],
                "largest_group_share": result["report"][
                    "largest_group_share"
                ],
                "manifest_sha256": result["manifest_sha256"],
                "report_sha256": result["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["report"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
