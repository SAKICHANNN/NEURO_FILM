#!/usr/bin/env python3
"""Run the exact U6.P4R TIFF integrity/decode/visual preflight."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_uniform_grain_preflight import (  # noqa: E402
    run_preflight,
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
    output = ROOT / "outputs/experiments/u6_p4r_uniform_grain_preflight_v1"
    report = run_preflight(
        root=ROOT,
        source_config=source,
        analysis_config=analysis,
        output_dir=output,
    )
    digest = write_report(report, output / "report.json")
    print(
        json.dumps(
            {
                "source_count": report["source_count"],
                "total_bytes": report["total_bytes"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
