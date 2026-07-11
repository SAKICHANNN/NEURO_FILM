#!/usr/bin/env python3
"""Create an ignored blinded FilmCase review sheet and its private mapping."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmcase.vision_audit import VisionAuditError, build_blind_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a three-round blinded FilmCase review plan.")
    parser.add_argument("--samples", required=True, help="Comma-separated frozen sample IDs.")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate IDs; never shown in the public sheet.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "filmcase" / "u42_blind_audit")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_blind_audit(args.samples.split(","), args.candidates.split(","), seed=args.seed)
    except VisionAuditError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "review_sheet.json").write_text(json.dumps(list(plan.sheet), indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "PRIVATE_label_mapping.json").write_text(json.dumps(list(plan.mapping), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": 3, "sheet_rows": len(plan.sheet), "mapping_rows": len(plan.mapping), "output_dir": str(args.output_dir) if args.write else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
