#!/usr/bin/env python
"""Extract and audit the frozen Kodak 250D channel MTF graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_kodak_250d_2383_datasheet_graphs import extract
from src.eval.physical_mtf_source import audit_source, load_contract


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
        default=ROOT / "configs/u6_p5i_kodak_250d_mtf_source_v1.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="re-extract all frozen graph rasters before auditing (requires pypdf)",
    )
    args = parser.parse_args()
    config = load_contract(args.config)
    if args.extract:
        extract(ROOT / "outputs/u5_r2aa0_source_audit/extracted")
    report = audit_source(config, ROOT, overlay_path=args.overlay)
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "source_pass": report["source_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
