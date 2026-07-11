#!/usr/bin/env python3
"""Create a review-only contact sheet from public anonymous FilmCase assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a contact sheet from a public anonymous FilmCase review sheet.")
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-width", type=int, default=220)
    args = parser.parse_args()
    rows = json.loads(args.review_sheet.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("review sheet must be a list")
    rows = [row for row in rows if row.get("round") == args.round]
    if not rows or any(not isinstance(row.get("assets"), dict) for row in rows):
        raise ValueError("review sheet needs anonymous assets for the requested round")
    labels = rows[0]["labels"]
    if any(row["labels"] != labels for row in rows):
        raise ValueError("inconsistent label layout")
    header, gap = 24, 4
    cell_width = args.cell_width
    cell_height = int(cell_width * 0.75)
    canvas = Image.new("RGB", (120 + len(labels) * (cell_width + gap), header + len(rows) * (cell_height + header + gap)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((120 + column * (cell_width + gap) + 4, 4), label, fill="white")
    root = args.review_sheet.parent
    for row_index, row in enumerate(rows):
        y = header + row_index * (cell_height + header + gap)
        draw.text((4, y + 4), str(row["sample_id"]), fill="white")
        for column, label in enumerate(labels):
            asset = root / row["assets"][label]
            with Image.open(asset) as image:
                cell = ImageOps.fit(image.convert("RGB"), (cell_width, cell_height), method=Image.Resampling.LANCZOS)
            canvas.paste(cell, (120 + column * (cell_width + gap), y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
