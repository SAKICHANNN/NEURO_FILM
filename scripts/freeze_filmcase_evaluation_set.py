#!/usr/bin/env python3
"""Freeze a local union-source seed set without rendering or copying images."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmcase.evaluation import EvaluationContractError, coverage_report, freeze_union_sources

DEFAULT_SOURCE = ROOT / "outputs" / "contact_sheets" / "velvia50_union_all_models_20260616" / "UNION_SOURCES.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "filmcase" / "u41_provisional_eval_set.json"
DEFAULT_GOLD = ("01", "05", "08", "09", "11", "18", "21", "29")
BUCKETS = {"01": ["fine_detail", "deep_shadow"], "05": ["foliage", "saturated_objects"], "08": ["sky", "high_key"], "09": ["high_key", "saturated_objects"], "11": ["deep_shadow", "saturated_objects"], "18": ["fine_detail", "text_logo"], "21": ["fine_detail"], "29": ["sky", "high_key"]}

def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a provisional FilmCase artifact-evaluation seed set.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-ids", default=",".join(DEFAULT_GOLD))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        frozen = freeze_union_sources(args.source, root=ROOT, gold_ids=args.gold_ids.split(","), bucket_map=BUCKETS)
    except EvaluationContractError as exc:
        print(f"ERROR: {exc}")
        return 2
    report = coverage_report(frozen)
    payload = {"frozen_set": frozen, "coverage": report}
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": frozen["status"], "coverage": report, "output": str(args.output) if args.write else None}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
