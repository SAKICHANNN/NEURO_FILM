#!/usr/bin/env python
"""Build a worst-alignment sheet from the passing fresh FiveK audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps


def _tile(path: Path, size: tuple[int, int]) -> Image.Image:
    data = tifffile.imread(path)
    pixels = np.rint(
        np.clip(data.astype(np.float32) / 65535.0, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return ImageOps.contain(Image.fromarray(pixels, mode="RGB"), size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("automatic_pass") is not True:
        raise SystemExit("alignment sheet requires automatic passage")
    candidates = sorted(
        manifest["rows"],
        key=lambda row: (
            float(row["gradient_correlation"]),
            -float(row["phase_shift_pixels_at_audit_scale"]),
            row["pair_id"],
        ),
    )
    selected: list[dict] = []
    seen_cameras: set[str] = set()
    for row in [*candidates, *candidates]:
        if len(selected) < int(args.rows) // 2:
            selected.append(row)
            seen_cameras.add(row["camera_model"])
        elif row["camera_model"] not in seen_cameras:
            selected.append(row)
            seen_cameras.add(row["camera_model"])
        if len(selected) == int(args.rows):
            break
    tile_size = (420, 280)
    header = 32
    sheet = Image.new(
        "RGB",
        (
            tile_size[0] * 2,
            (tile_size[1] + header) * len(selected),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        top = index * (tile_size[1] + header)
        label = (
            f"{row['source_name']} | corr={row['gradient_correlation']:.3f} "
            f"shift={row['phase_shift_pixels_at_audit_scale']:.3f}px"
        )
        draw.text((4, top + 4), f"source | {label}", fill="black")
        draw.text((tile_size[0] + 4, top + 4), "filtered target", fill="black")
        for column, key in enumerate(("source_path", "target_path")):
            tile = _tile(Path(row[key]), tile_size)
            left = column * tile_size[0]
            sheet.paste(
                tile,
                (
                    left + (tile_size[0] - tile.width) // 2,
                    top + header + (tile_size[1] - tile.height) // 2,
                ),
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "alignment_sheet.png"
    sheet.save(output, format="PNG", compress_level=6)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
