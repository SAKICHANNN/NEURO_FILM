"""Run the frozen U5.R2AY1 neutral-base plus fixed-AO6 ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fivek_neutral_base_fixed_ao6_ablation import (  # noqa: E402
    run_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u5_r2ay1_fivek_neutral_base_fixed_ao6_ablation_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = ROOT
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    software_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    result = run_ablation(
        root=root,
        config=config,
        config_path=config_path,
        output_dir=root / args.output_dir,
        software_commit=software_commit,
    )
    print(
        json.dumps(
            {
                "automatic_pass": result["report"]["automatic_pass"],
                "report_sha256": result["report_sha256"],
                "stable_evidence_id": result["report"]["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
