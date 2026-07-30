from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_residual_fresh_confirmation import (
    render_and_evaluate,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u5_r2az1_density_residual_fresh_confirmation_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / args.output_dir
    report = render_and_evaluate(
        root=ROOT, config=config, output_dir=output_dir
    )
    report_sha256 = write_json(report, output_dir / "report.json")
    print(
        json.dumps(
            {**report, "report_sha256": report_sha256},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
