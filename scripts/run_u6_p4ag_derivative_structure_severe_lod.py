#!/usr/bin/env python
"""Run the frozen U6.P4AG synthetic severe/LOD audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_derivative_structure_severe_lod import (  # noqa: E402
    evaluate_severe_lod,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u6_p4ag_derivative_structure_severe_lod_v1.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_severe_lod(load_contract(args.config), ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(json.dumps({key: report[key] for key in ("decision_key", "severe_pass", "lod_pass", "stable_evidence_id")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
