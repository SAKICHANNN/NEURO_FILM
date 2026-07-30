from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_residual_adjudication import (
    adjudicate_density_residual_visual,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u5_r2az0v_density_residual_visual_v1.json",
    )
    parser.add_argument(
        "--observations",
        default=(
            "configs/"
            "u5_r2az0v_density_residual_visual_observations_v1.json"
        ),
    )
    parser.add_argument(
        "--build-dir",
        default="outputs/u5_r2az0v_density_residual_visual_v1",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/u5_r2az0v_density_residual_visual_v1/"
            "adjudication.json"
        ),
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    report = adjudicate_density_residual_visual(
        root=ROOT,
        config=config,
        observations_path=ROOT / args.observations,
        build_dir=ROOT / args.build_dir,
    )
    report_sha256 = write_json(report, ROOT / args.output)
    print(
        json.dumps(
            {**report, "report_sha256": report_sha256},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
