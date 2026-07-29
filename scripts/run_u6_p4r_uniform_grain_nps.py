#!/usr/bin/env python3
"""Run the admitted U6.P4R scanner-convolved NPS/ACF audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_uniform_grain_analysis import (  # noqa: E402
    run_analysis,
    write_report,
)


def main() -> None:
    source = json.loads(
        (ROOT / "configs/u6_p4r_uniform_grain_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    analysis = json.loads(
        (
            ROOT / "configs/u6_p4r_uniform_grain_nps_feasibility_v1.json"
        ).read_text(encoding="utf-8")
    )
    execution = json.loads(
        (
            ROOT / "configs/u6_p4r_uniform_grain_nps_execution_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = run_analysis(
        root=ROOT,
        source_config=source,
        analysis_config=analysis,
        execution_config=execution,
    )
    output = (
        ROOT
        / "outputs/experiments/u6_p4r_uniform_grain_nps_v1/report.json"
    )
    digest = write_report(report, output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
