"""Render every frozen CT8 automatic trigger for supplemental visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.ct5_data import CT5DataContract, load_ct5_working_image  # noqa: E402
from src.roll2film.ct5_fullres import linear_to_u8  # noqa: E402
from src.roll2film.ct8_final import (  # noqa: E402
    load_final_manifest,
    operator_from_frozen_bundle,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct8_trigger_review.json",
    )
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs" / "roll2film_ct8_final_policy.json"
    )
    parser.add_argument(
        "--ct5-config", type=Path, default=ROOT / "configs" / "roll2film_ct5_baselines.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "filmset_evidence" / "final_628_lockbox.jsonl",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct8_final_628" / "report.json",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "pilot_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct8_final_628" / "trigger_review",
    )
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(linear_to_u8(values), "RGB").save(path, "PNG", optimize=True)


def _make_page(rows: list[dict[str, Any]], output: Path, tile_size: int) -> None:
    columns = ("input", "target", "best_basic", "primary")
    header = 24
    sheet = Image.new("RGB", (tile_size * 4, header + tile_size * len(rows)), "#151515")
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(columns):
        draw.text((column * tile_size + 5, 5), label, fill="white")
    for row_index, row in enumerate(rows):
        for column, label in enumerate(columns):
            with Image.open(row[label]) as image:
                tile = image.convert("RGB")
                tile.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_size, tile_size), "black")
                canvas.paste(tile, ((tile_size - tile.width) // 2, (tile_size - tile.height) // 2))
            sheet.paste(canvas, (column * tile_size, header + row_index * tile_size))
        draw.text((3, header + row_index * tile_size + 3), row["content_id"], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=92, subsampling=0)


def main() -> int:
    args = parse_args()
    review = json.loads(args.review_config.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if sha256_file(args.source_report) != review["source_report_sha256"]:
        raise ValueError("source CT8 report hash mismatch")
    if sha256_file(args.policy) != review["source_policy_sha256"]:
        raise ValueError("source CT8 policy hash mismatch")
    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if report["decision_state"] != "awaiting_original_resolution_visual_severe_adjudication":
        raise ValueError("CT8 report is not awaiting the frozen visual adjudication")
    expected = policy["expected"]
    by_content = load_final_manifest(
        args.manifest,
        expected_sha256=policy["source_contract"]["final_628_manifest_sha256"],
        expected_identities=int(expected["identities"]),
        expected_rows=int(expected["manifest_rows"]),
        domains=expected["domains"],
    )
    pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    contract = CT5DataContract.from_config(args.ct5_config, ROOT)
    output_dir = args.output_dir.resolve()
    rendered: dict[str, Any] = {}
    all_pages: list[Path] = []
    for domain in expected["domains"]:
        trigger_ids = report["domains"][domain]["automatic_trigger_content_ids"]
        names = policy["frozen_recipe_bank"][domain]
        primary = operator_from_frozen_bundle(pilot["operator_bundles"][domain][names["primary"]])
        basic = operator_from_frozen_bundle(pilot["operator_bundles"][domain][names["best_basic"]])
        review_rows: list[dict[str, Any]] = []
        for index, content_id in enumerate(trigger_ids, 1):
            rows = by_content[content_id]
            source = load_ct5_working_image(contract, rows["input"]).pixels
            target = load_ct5_working_image(contract, rows[domain]).pixels
            primary_values = primary.apply(source.reshape(-1, 3)).reshape(source.shape)
            basic_values = basic.apply(source.reshape(-1, 3)).reshape(source.shape)
            case_dir = output_dir / "cases" / domain / content_id.replace(":", "_")
            paths = {
                "input": case_dir / "input.png",
                "target": case_dir / "target.png",
                "best_basic": case_dir / f"best_basic_{names['best_basic']}.png",
                "primary": case_dir / f"primary_{names['primary']}.png",
            }
            _save(paths["input"], source)
            _save(paths["target"], target)
            _save(paths["best_basic"], basic_values)
            _save(paths["primary"], primary_values)
            review_rows.append({"content_id": content_id, **paths})
            if index % 25 == 0 or index == len(trigger_ids):
                print(f"CT8 trigger review {domain} {index}/{len(trigger_ids)}", flush=True)
        pages: list[Path] = []
        rows_per_sheet = int(review["rows_per_sheet"])
        for page_index, start in enumerate(range(0, len(review_rows), rows_per_sheet), 1):
            page = output_dir / "sheets" / f"{domain}__page_{page_index:02d}.jpg"
            _make_page(review_rows[start : start + rows_per_sheet], page, int(review["tile_size"]))
            pages.append(page)
            all_pages.append(page)
        rendered[domain] = {
            "trigger_count": len(trigger_ids),
            "content_ids": trigger_ids,
            "pages": [str(path) for path in pages],
        }
    result = {
        "schema_version": 1,
        "experiment_id": review["experiment_id"],
        "review_config_sha256": sha256_file(args.review_config),
        "source_report_sha256": sha256_file(args.source_report),
        "software_commit": _commit(),
        "domains": rendered,
        "sheet_sha256": {str(path): sha256_file(path) for path in all_pages},
        "visual_decision": "pending",
        "claim_boundary": review["claim_boundary"],
    }
    result_path = output_dir / "report.json"
    result_path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"report": str(result_path), "pages": len(all_pages)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
