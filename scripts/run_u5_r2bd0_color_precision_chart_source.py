#!/usr/bin/env python
"""Inventory the frozen Color Precision chart archive without extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.color_precision_chart_source import inspect_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bd0_color_precision_chart_source_v1.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "D:/neuro_film_external/"
            "u5_r2bd0_color_precision_chart_source_v1/Charts-PDFs.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2bd0_color_precision_chart_source_v1/"
            "archive_report.json"
        ),
    )
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = inspect_archive(args.archive, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive_bytes": report["archive_bytes"],
                "archive_sha256": report["archive_sha256"],
                "member_count": report["member_count"],
                "extension_counts": report["extension_counts"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
