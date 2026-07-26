"""Integrity and presentation helpers for U5.R2L0 visual review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.eval.density_strength_oracle import (
    DensityStrengthOracleError,
    sha256_file,
)


def crop_boxes(width: int, height: int, size: int = 192) -> list[tuple[int, int, int, int]]:
    """Return deterministic corner/centre crops without scaling."""

    edge = min(size, width, height)
    x_max = width - edge
    y_max = height - edge
    centres = (
        (0, 0),
        (x_max, 0),
        (x_max // 2, y_max // 2),
        (0, y_max),
        (x_max, y_max),
    )
    return [(x, y, x + edge, y + edge) for x, y in centres]


def _overview(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        contained = ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return canvas


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DensityStrengthOracleError("review input must be a JSON object")
    return value


def build_review_records(
    *,
    root: Path,
    oracle_config: Mapping[str, Any],
    selected_manifest_path: Path,
    e1_config_path: Path,
) -> list[dict[str, Any]]:
    """Verify sources and candidate outputs, then bind exact review records."""

    selected = _load_json(selected_manifest_path)
    parent_report_path = root / str(
        oracle_config["parent"]["automatic_report"]
    )
    parent_report = _load_json(parent_report_path)
    if sha256_file(e1_config_path) != str(parent_report["config_sha256"]):
        raise DensityStrengthOracleError("E1 config hash mismatch")
    e1_config = _load_json(e1_config_path)
    frozen_path = root / str(e1_config["frozen_set"])
    if sha256_file(frozen_path) != str(e1_config["frozen_set_sha256"]):
        raise DensityStrengthOracleError("frozen-set file hash mismatch")
    frozen = _load_json(frozen_path)
    samples = frozen.get("frozen_set", frozen).get("samples", [])
    sources = {str(row["id"]): dict(row) for row in samples}

    parent_manifest_path = root / str(
        oracle_config["parent"]["render_manifest"]
    )
    parent_manifest = _load_json(parent_manifest_path)
    baseline_id = str(oracle_config["baseline_candidate_id"])
    baseline = {
        str(row["sample_id"]): dict(row)
        for row in parent_manifest["records"]
        if str(row["candidate_id"]) == baseline_id
    }
    records = []
    for row in selected["records"]:
        sample_id = str(row["sample_id"])
        if sample_id not in sources or sample_id not in baseline:
            raise DensityStrengthOracleError(
                f"review membership mismatch: {sample_id}"
            )
        source = sources[sample_id]
        source_path = (root / str(source["source_path"])).resolve()
        selected_path = (root / str(row["selected_output"])).resolve()
        baseline_path = (
            parent_manifest_path.parent / str(baseline[sample_id]["output"])
        ).resolve()
        hashes = (
            (source_path, str(source["source_sha256"]), "source"),
            (
                selected_path,
                str(row["selected_output_sha256"]),
                "selected output",
            ),
            (
                baseline_path,
                str(baseline[sample_id]["output_sha256"]),
                "baseline output",
            ),
        )
        for path, expected, label in hashes:
            if sha256_file(path) != expected:
                raise DensityStrengthOracleError(
                    f"{label} hash mismatch: {sample_id}"
                )
        with Image.open(selected_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        records.append(
            {
                "sample_id": sample_id,
                "split": str(row["split"]),
                "selected_strength": float(row["selected_strength"]),
                "source_path": source_path,
                "baseline_path": baseline_path,
                "selected_path": selected_path,
                "selected_output_sha256": str(row["selected_output_sha256"]),
                "size": [width, height],
                "crop_boxes": crop_boxes(width, height),
            }
        )
    if len(records) != int(oracle_config["expected_samples"]):
        raise DensityStrengthOracleError("review record count drift")
    return records


def write_review_pages(
    records: list[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write overview and 1:1 five-crop pages for every selected output."""

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    overview_pages = []
    crop_pages = []
    for page_index in range(0, len(records), 7):
        subset = records[page_index : page_index + 7]
        overview = Image.new("RGB", (1200, 7 * 224 + 36), "#dedede")
        crop_page = Image.new("RGB", (1450, 7 * 204 + 36), "#dedede")
        overview_draw = ImageDraw.Draw(overview)
        crop_draw = ImageDraw.Draw(crop_page)
        title = f"U5.R2L0 review page {page_index // 7 + 1}"
        overview_draw.text((8, 8), title, fill="black", font=font)
        crop_draw.text(
            (8, 8),
            f"{title} / selected output 1:1 crops TL TR C BL BR",
            fill="black",
            font=font,
        )
        for row_index, row in enumerate(subset):
            y_overview = 36 + row_index * 224
            label = (
                f"{row['sample_id']} {row['split']} "
                f"s{float(row['selected_strength']):.2f}"
            )
            paths = (
                ("source", Path(row["source_path"])),
                ("fixed s0.50", Path(row["baseline_path"])),
                ("selected", Path(row["selected_path"])),
            )
            for column, (column_label, path) in enumerate(paths):
                x = column * 400
                tile = _overview(path, (392, 196))
                overview.paste(tile, (x + 4, y_overview + 20))
                overview_draw.text(
                    (x + 8, y_overview + 3),
                    f"{label} / {column_label}",
                    fill="black",
                    font=font,
                )

            y_crop = 36 + row_index * 204
            crop_draw.text((4, y_crop + 3), label, fill="black", font=font)
            with Image.open(Path(row["selected_path"])) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                for crop_index, box in enumerate(row["crop_boxes"]):
                    crop = rgb.crop(tuple(box))
                    tile = Image.new("RGB", (192, 192), "white")
                    tile.paste(
                        crop,
                        (
                            (192 - crop.width) // 2,
                            (192 - crop.height) // 2,
                        ),
                    )
                    x = 250 + crop_index * 238
                    crop_page.paste(tile, (x, y_crop + 10))
                    crop_draw.text(
                        (x, y_crop),
                        ("TL", "TR", "C", "BL", "BR")[crop_index],
                        fill="black",
                        font=font,
                    )
        overview_path = output_dir / f"overview_{page_index // 7 + 1:02d}.png"
        crop_path = output_dir / f"crops_1to1_{page_index // 7 + 1:02d}.png"
        overview.save(overview_path, format="PNG", compress_level=6)
        crop_page.save(crop_path, format="PNG", compress_level=6)
        overview_pages.append(
            {
                "path": str(overview_path.resolve()),
                "sha256": sha256_file(overview_path),
                "sample_ids": [str(row["sample_id"]) for row in subset],
            }
        )
        crop_pages.append(
            {
                "path": str(crop_path.resolve()),
                "sha256": sha256_file(crop_path),
                "sample_ids": [str(row["sample_id"]) for row in subset],
            }
        )
    return {
        "schema_version": 1,
        "overview_pages": overview_pages,
        "one_to_one_crop_pages": crop_pages,
        "reviewed_sample_count": len(records),
        "coverage": "full overview plus five unscaled output crops per sample",
    }


def evidence_sha256(value: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
