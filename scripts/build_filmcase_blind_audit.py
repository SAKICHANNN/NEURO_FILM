#!/usr/bin/env python3
"""Create an ignored blinded FilmCase review sheet and its private mapping."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmcase.vision_audit import VisionAuditError, build_blind_audit


def load_render_records(path: Path) -> dict[tuple[str, str], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise VisionAuditError(f"{path} does not contain render records")
    result: dict[tuple[str, str], Path] = {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str) or row.get("sample_id") is None or not isinstance(row.get("output"), str):
            raise VisionAuditError("render record requires candidate_id, sample_id and output")
        key = (row["candidate_id"], str(row["sample_id"]))
        source = ROOT / row["output"]
        if key in result or not source.is_file():
            raise VisionAuditError(f"missing or duplicate render record: {key}")
        result[key] = source
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a three-round blinded FilmCase review plan.")
    parser.add_argument("--samples", required=True, help="Comma-separated frozen sample IDs.")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate IDs; never shown in the public sheet.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--render-manifest", type=Path, help="Normalized replay manifest. When supplied, write anonymous image copies for the public sheet.")
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
        sheet = [dict(row) for row in plan.sheet]
        if args.render_manifest:
            try:
                renders = load_render_records(args.render_manifest)
            except VisionAuditError as exc:
                print(f"ERROR: {exc}")
                return 2
            sheet_by_key = {(row["round"], row["sample_id"]): row for row in sheet}
            for mapping in plan.mapping:
                key = (mapping["round"], mapping["sample_id"])
                asset_paths: dict[str, str] = {}
                for label, candidate in mapping["label_to_candidate"].items():
                    source = renders.get((candidate, mapping["sample_id"]))
                    if source is None:
                        print(f"ERROR: no render for {candidate}/{mapping['sample_id']}")
                        return 2
                    destination = args.output_dir / "assets" / f"round_{mapping['round']}" / f"sample_{mapping['sample_id']}" / f"{label}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    asset_paths[label] = str(destination.relative_to(args.output_dir)).replace("\\", "/")
                sheet_by_key[key]["assets"] = asset_paths
        (args.output_dir / "review_sheet.json").write_text(json.dumps(sheet, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "PRIVATE_label_mapping.json").write_text(json.dumps(list(plan.mapping), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": 3, "sheet_rows": len(plan.sheet), "mapping_rows": len(plan.mapping), "output_dir": str(args.output_dir) if args.write else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
