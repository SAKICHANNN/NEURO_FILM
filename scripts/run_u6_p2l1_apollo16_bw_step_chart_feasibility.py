"""Run the frozen P2L1 AS16-111 wedge source-geometry audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.apollo16_bw_step_chart_feasibility import (  # noqa: E402
    evaluate_apollo16_bw_step_chart_feasibility,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/u6_p2l1_apollo16_bw_step_chart_feasibility_v1.json"
        ),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_apollo16_bw_step_chart_feasibility(config, ROOT)
    path = ROOT / str(config["report"])
    write_report(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
