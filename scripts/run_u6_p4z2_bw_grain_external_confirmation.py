#!/usr/bin/env python3
"""Run the frozen U6.P4Z2 B&W grain external confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bw_grain_external_confirmation import (  # noqa: E402
    evaluate_external_confirmation,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", choices=("a", "b"), default="a")
    args = parser.parse_args()
    path = ROOT / "configs/u6_p4z2_bw_grain_external_confirmation_v1.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    report = evaluate_external_confirmation(root=ROOT, contract=contract)
    output = ROOT / contract["outputs"][f"report_{args.output}"]
    digest = write_report(report, output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": digest,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
