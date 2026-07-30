#!/usr/bin/env python
"""Run the frozen AY5 hard case/medoid development experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_hard_case_medoid_development import (  # noqa: E402
    run_development,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay5_hard_case_medoid_development_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = run_development(
        root=ROOT,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        software_commit=commit,
    )
    print(
        json.dumps(
            {
                "automatic_pass": result["report"]["automatic_pass"],
                "selected_method": result["report"]["selected_method"],
                "report_sha256": result["report_sha256"],
                "stable_evidence_id": result["report"][
                    "stable_evidence_id"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
