#!/usr/bin/env python
"""Run the frozen U6.P5L measured-MTF target-grid LOD compiler."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_measured_mtf_lod import compile_and_evaluate_lod, load_contract


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
        default=ROOT / "configs/u6_p5l_measured_mtf_lod_compiler_v1.json",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    bundle, report = compile_and_evaluate_lod(
        load_contract(args.config), ROOT, diagnostic_path=args.diagnostic
    )
    _write_json(args.bundle, bundle)
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
                "lod_bundle_id": bundle["lod_bundle_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
