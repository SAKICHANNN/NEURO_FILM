"""Render full-resolution RF1.4B1 held-out comparisons and roll contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.gold_transform_consistency import (  # noqa: E402
    GoldTransformConsistencyError,
    operator_from_dict,
)

PANEL_KEYS = (
    ("identity", "SOURCE"),
    ("bounded_per_channel_affine", "CHANNEL AFFINE"),
    ("bounded_ridge_3x3_affine", "3x3 AFFINE"),
    ("seplut17_monotone_plus_bounded_ridge_3x3", "SEPLUT17 + 3x3"),
    ("target", "DISPLAY PROXY TARGET"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb8(values: np.ndarray) -> Image.Image:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


def _save_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp" + path.suffix)
    image.save(temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "real_film_gold_transform_consistency_v1.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf1_4b1" / "report.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "real_film" / "stock_pilots_v1" / "rf1_4b1" / "visual",
    )
    args = parser.parse_args()
    config = _load(args.config)
    report = _load(args.report)
    if not report["metric_passed"]:
        raise GoldTransformConsistencyError("visual rendering is forbidden before metric pass")
    if report["config_sha256"] != _sha(args.config):
        raise GoldTransformConsistencyError("report/config hash mismatch")
    alignment_report_path = ROOT / config["alignment_report"]
    if _sha(alignment_report_path) != config["alignment_report_sha256"]:
        raise GoldTransformConsistencyError("pinned alignment report hash mismatch")
    pairs = {row["frame_id"]: row for row in _load(alignment_report_path)["pair_records"]}

    strips_by_roll: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    manifest_files: list[dict[str, str]] = []
    for fold in report["fold_results"]:
        operators = {key: operator_from_dict(value) for key, value in fold["operators"].items()}
        for frame_metric in fold["metrics"]["identity"]["per_frame"]:
            frame_id = frame_metric["frame_id"]
            pair = pairs[frame_id]
            preview_path = ROOT / config["download_root"] / Path(*Path(pair["preview_path"]).parts)
            target_path = ROOT / config["download_root"] / Path(*Path(pair["proxy_path"]).parts)
            with Image.open(preview_path) as image:
                x0, y0, x1, y1 = (int(value) for value in pair["bbox"])
                source_image = ImageOps.exif_transpose(image).convert("RGB").crop((x0, y0, x1, y1))
            with Image.open(target_path) as image:
                target_image = ImageOps.exif_transpose(image).convert("RGB")
            source = np.asarray(source_image, dtype=np.float64) / 255.0
            height, width = source.shape[:2]
            rendered: dict[str, Image.Image] = {"identity": source_image, "target": target_image}
            for key, operator in operators.items():
                if key != "identity":
                    rendered[key] = _rgb8(operator.apply(source.reshape(-1, 3)).reshape(height, width, 3))

            header = 28
            strip = Image.new("RGB", (width * len(PANEL_KEYS), height + header), "white")
            draw = ImageDraw.Draw(strip)
            for column, (key, label) in enumerate(PANEL_KEYS):
                strip.paste(rendered[key], (column * width, header))
                draw.text((column * width + 6, 7), label, fill="black")
            strip_path = args.output_dir / "full_resolution_strips" / f"{frame_id}.png"
            _save_atomic(strip, strip_path)
            strips_by_roll[str(pair["roll_id"])].append((frame_id, strip_path))
            manifest_files.append({"kind": "full_resolution_strip", "path": str(strip_path.relative_to(ROOT)), "sha256": _sha(strip_path)})

    for roll_id, rows in sorted(strips_by_roll.items()):
        thumb_width = 260
        label_height = 24
        thumbs: list[tuple[str, Image.Image]] = []
        for frame_id, strip_path in sorted(rows):
            with Image.open(strip_path) as strip:
                thumb_height = max(1, round(strip.height * (thumb_width * len(PANEL_KEYS)) / strip.width))
                thumbs.append((frame_id, strip.resize((thumb_width * len(PANEL_KEYS), thumb_height), Image.Resampling.LANCZOS)))
        sheet_width = thumb_width * len(PANEL_KEYS)
        sheet_height = sum(image.height + label_height for _, image in thumbs)
        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for frame_id, thumb in thumbs:
            draw.text((6, y + 5), f"{roll_id} / {frame_id}", fill="black")
            y += label_height
            sheet.paste(thumb, (0, y))
            y += thumb.height
        sheet_path = args.output_dir / "roll_contact_sheets" / f"{roll_id}.jpg"
        _save_atomic(sheet, sheet_path)
        manifest_files.append({"kind": "roll_contact_sheet", "path": str(sheet_path.relative_to(ROOT)), "sha256": _sha(sheet_path)})

    manifest = {
        "schema_version": 1,
        "experiment_id": report["experiment_id"],
        "source_report_sha256": _sha(args.report),
        "panels": [label for _, label in PANEL_KEYS],
        "full_resolution_strips": sum(row["kind"] == "full_resolution_strip" for row in manifest_files),
        "roll_contact_sheets": sum(row["kind"] == "roll_contact_sheet" for row in manifest_files),
        "files": manifest_files,
    }
    manifest_path = args.output_dir / "manifest.json"
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "full_resolution_strips": manifest["full_resolution_strips"],
        "roll_contact_sheets": manifest["roll_contact_sheets"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
