#!/usr/bin/env python3
"""Run the frozen BL12 SA-LUT compatibility preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.salut_reference_lut_preflight import run_salut_preflight  # noqa: E402


CONFIG = ROOT / "configs/u5_r2bl12_salut_reference_lut_preflight_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = run_salut_preflight(
        root=ROOT,
        config=json.loads(CONFIG.read_text(encoding="utf-8")),
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
