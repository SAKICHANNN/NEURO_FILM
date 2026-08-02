#!/usr/bin/env python
"""Run the frozen U5.R2BU2 VISION3 granularity-diversity audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_vision3_granularity_diversity import (
    audit_diversity,
    load_contract,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bu2_kodak_vision3_granularity_diversity_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_diversity(
        load_contract(args.config), ROOT, overlay_dir=args.overlay_dir
    )
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "diversity_pass": report["diversity_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
