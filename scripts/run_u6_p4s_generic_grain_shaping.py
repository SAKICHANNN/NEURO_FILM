#!/usr/bin/env python3
"""Run the frozen U6.P4S generic grain-shaping comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_uniform_grain_shaping import (  # noqa: E402
    run_generic_grain_shaping,
    write_report,
)


def main() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4s_generic_grain_shaping_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = run_generic_grain_shaping(root=ROOT, contract=contract)
    output = (
        ROOT
        / "outputs/experiments/u6_p4s_generic_grain_shaping_v1/report.json"
    )
    digest = write_report(report, output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_sigma_yx": report[
                    "selected_shared_sigma_yx_pixels"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
