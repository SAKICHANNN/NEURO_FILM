#!/usr/bin/env python3
"""Run the frozen BL13 AO6 non-basic chroma emphasis experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_nonbasic_chroma_emphasis import (  # noqa: E402
    run_ao6_nonbasic_chroma_emphasis,
)


CONFIG = ROOT / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    os.environ["OMP_NUM_THREADS"] = "1"
    report = run_ao6_nonbasic_chroma_emphasis(
        root=ROOT,
        config=json.loads(CONFIG.read_text(encoding="utf-8")),
        config_path=CONFIG,
        output_dir=output_dir,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    print(
        json.dumps(
            {
                "automatic_gate_pass": report["automatic_gate_pass"],
                "aggregate": report["aggregate"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
