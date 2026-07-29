#!/usr/bin/env python
"""Run the frozen U6.P3L RAW scene-linear backing-return audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.scene_linear_backing_return import (  # noqa: E402
    evaluate_scene_linear_backing_return,
    load_contract,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p3l_scene_linear_backing_return_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = evaluate_scene_linear_backing_return(
        root=ROOT, contract=contract
    )
    digest = write_report(report, args.output)
    print(
        json.dumps(
            {
                "report_sha256": digest,
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_pass": report["automatic_pass"],
                "branch": report["branch"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
