#!/usr/bin/env python3
"""Run the frozen U6.P4T physical-domain and held-ACF audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.real_uniform_grain_physical import (  # noqa: E402
    run_anisotropic_grain_physical,
    write_report,
)


def main() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4t_anisotropic_grain_physical_v1.json"
        ).read_text(encoding="utf-8")
    )
    output_dir = (
        ROOT / "outputs/experiments/u6_p4t_anisotropic_grain_physical_v1"
    )
    report = run_anisotropic_grain_physical(
        root=ROOT,
        contract=contract,
        output_dir=output_dir,
    )
    digest = write_report(report, output_dir / "report.json")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "automatic_pass": report["automatic_pass"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
