"""Run the frozen P2K step-chart layout and geometry preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.apollo_step_chart_preflight import (  # noqa: E402
    evaluate_apollo_step_chart_preflight,
    render_preview,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2k_apollo_step_chart_preflight_v1.json",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report, sampled = evaluate_apollo_step_chart_preflight(config, ROOT)
    report_path = args.report or ROOT / str(config["outputs"]["report"])
    preview_path = args.preview or ROOT / str(config["outputs"]["preview"])
    render_preview(sampled, preview_path)
    report_sha256 = write_report(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": report_sha256,
                "preview": str(preview_path),
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
