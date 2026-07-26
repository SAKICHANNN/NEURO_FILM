#!/usr/bin/env python
"""Build the deterministic SF2.7A visual alignment evidence sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.scanner_nuisance import (  # noqa: E402
    _native_rgb,
    align_source_to_scan,
)


def _grid_segments(grid: dict[str, object]) -> list[np.ndarray]:
    columns = int(grid["columns"])
    rows = int(grid["rows"])
    left = float(grid["left_edge_px"])
    right = float(grid["right_edge_px"])
    top = float(grid["top_edge_px"])
    bottom = float(grid["bottom_edge_px"])
    segments = []
    for column in range(columns + 1):
        x = left + (right - left) * column / columns
        segments.append(np.float32([[[x, top], [x, bottom]]]))
    for row in range(rows + 1):
        y = top + (bottom - top) * row / rows
        segments.append(np.float32([[[left, y], [right, y]]]))
    return segments


def _preview(
    scan: np.ndarray,
    homography: np.ndarray,
    grid: dict[str, object],
) -> Image.Image:
    encoded = np.rint(np.clip(scan, 0.0, 1.0) * 255.0).astype(np.uint8)
    for segment in _grid_segments(grid):
        mapped = cv2.perspectiveTransform(segment, homography).reshape(-1, 2)
        cv2.line(
            encoded,
            tuple(np.rint(mapped[0]).astype(int)),
            tuple(np.rint(mapped[1]).astype(int)),
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    image = Image.fromarray(encoded, mode="RGB")
    image.thumbnail((360, 260), Image.Resampling.LANCZOS)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_7a_scanner_nuisance_quantification_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/sf2_7a_scanner_nuisance_quantification_v1/alignment_sheet.png",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / str(config["parent_audit"])).read_text(encoding="utf-8")
    )
    data_root = ROOT / str(config["data_root"])
    role_records = {
        str(record["role"]): record
        for record in parent["assets"]
        if str(record["role"]).startswith("nikon_")
    }
    cell_width = 360
    cell_height = 290
    sheet = Image.new(
        "RGB",
        (cell_width * len(config["pipeline_roles"]), cell_height * 5),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, role in enumerate(config["pipeline_roles"]):
        record = role_records[str(role)]
        with ZipFile(data_root / str(record["path"])) as archive:
            members = {
                str(member["slide_id"]): str(member["name"])
                for member in record["members"]
                if member.get("slide_id") is not None
            }
            for row, slide in enumerate(config["slide_ids"]):
                source = _native_rgb(
                    (data_root / "source" / f"slidescale{slide}.tif").read_bytes()
                )
                scan = _native_rgb(archive.read(members[str(slide)]))
                homography, diagnostics = align_source_to_scan(
                    source, scan, config["alignment"]
                )
                preview = _preview(scan, homography, config["source_grid"])
                x = column * cell_width
                y = row * cell_height
                sheet.paste(preview, (x, y + 24))
                label = (
                    f"{role} / slide {slide} / "
                    f"{diagnostics['ransac_inliers']} inliers / "
                    f"{diagnostics['median_inlier_reprojection_error_px']:.3f}px"
                )
                draw.text((x + 4, y + 4), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", compress_level=6)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
