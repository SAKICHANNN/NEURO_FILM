#!/usr/bin/env python3
"""Aggregate recorded FilmCase blind-audit reviews without generating scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmcase.vision_audit import BlindAuditPlan, VisionAuditError, aggregate_reviews


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VisionAuditError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise VisionAuditError(f"{path}:{line_number}: review must be an object")
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate recorded FilmCase blind-audit reviews.")
    parser.add_argument("--mapping", type=Path, required=True, help="Private label mapping JSON from build_filmcase_blind_audit.py.")
    parser.add_argument("--reviews", type=Path, required=True, help="Raw review JSONL; one object per label/round/sample.")
    parser.add_argument("--output", type=Path, required=True, help="Ignored report JSON path.")
    args = parser.parse_args()
    try:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        if not isinstance(mapping, list):
            raise VisionAuditError("mapping must be a list")
        plan = BlindAuditPlan(sheet=tuple(), mapping=tuple(mapping))
        report = aggregate_reviews(plan, read_jsonl(args.reviews))
    except (OSError, json.JSONDecodeError, VisionAuditError) as exc:
        print(f"ERROR: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed_count": report["reviewed_count"], "expected_review_count": report["expected_review_count"], "missing_review_count": report["missing_review_count"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
