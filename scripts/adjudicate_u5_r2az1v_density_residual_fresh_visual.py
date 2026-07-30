from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_residual_fresh_confirmation import write_json
from src.eval.density_residual_fresh_visual import adjudicate_fresh_visual


def main() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2az1_density_residual_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    observations = json.loads(
        (
            ROOT
            / "configs/"
            "u5_r2az1v_density_residual_fresh_visual_observations_v1.json"
        ).read_text(encoding="utf-8")
    )
    output_dir = (
        ROOT / "outputs/u5_r2az1v_density_residual_fresh_visual_v1"
    )
    report = adjudicate_fresh_visual(
        config=config,
        observations=observations,
        build_dir=output_dir,
    )
    report_sha256 = write_json(report, output_dir / "adjudication.json")
    print(
        json.dumps(
            {**report, "report_sha256": report_sha256},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
