#!/usr/bin/env python
"""Run the frozen U5.R2AY0 bounded neutral-base parameter pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_neutral_base_parameter_pilot import run_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ay0_fivek_neutral_base_parameter_pilot_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/u5_r2ay0_fivek_neutral_base_parameter_pilot_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = run_pilot(
        root=ROOT,
        config=config,
        config_path=config_path,
        output_dir=args.output.resolve(),
        software_commit=commit,
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "observed": report["observed"],
                "report_sha256": result["report_sha256"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
