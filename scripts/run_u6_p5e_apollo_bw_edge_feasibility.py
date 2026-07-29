"""Run the frozen P5E Apollo B&W edge-source feasibility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.apollo_bw_edge_feasibility import (  # noqa: E402
    evaluate_apollo_bw_edge_feasibility,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p5e_apollo_bw_edge_feasibility_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_apollo_bw_edge_feasibility(config, ROOT)
    write_report(ROOT / str(config["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
