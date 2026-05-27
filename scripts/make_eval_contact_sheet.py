#!/usr/bin/env python3
"""Create a before/after/diff contact sheet for render safety review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation contact sheets.")
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="CSV with before and after columns.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="render safety contact sheet")
    parser.add_argument("--max-pairs", type=int, default=40)
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def diff_image(before: Image.Image, after: Image.Image) -> Image.Image:
    before_arr = np.asarray(before, dtype=np.int16)
    after_arr = np.asarray(after, dtype=np.int16)
    diff = np.clip(np.abs(after_arr - before_arr) * 4, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def read_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, str]]:
    if args.before and args.after:
        return [(args.before, args.after, args.before.stem)]
    if not args.manifest:
        raise ValueError("Provide either --before/--after or --manifest")

    pairs: list[tuple[Path, Path, str]] = []
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            before_raw = row.get("before") or row.get("input") or row.get("source")
            after_raw = row.get("after") or row.get("output") or row.get("render")
            if before_raw and after_raw:
                pairs.append((Path(before_raw), Path(after_raw), row.get("id") or f"{index:02d}"))
            if len(pairs) >= args.max_pairs:
                break
    return pairs


def make_tile(before_path: Path, after_path: Path, label: str) -> Image.Image:
    font = ImageFont.load_default()
    before = load_rgb(before_path)
    after = load_rgb(after_path)
    if before.size != after.size:
        after = after.resize(before.size, Image.Resampling.BICUBIC)
    diff = diff_image(before, after)
    for image in (before, after, diff):
        image.thumbnail((220, 180), Image.Resampling.LANCZOS)
    tile_w = 220
    label_h = 34
    tile = Image.new("RGB", (tile_w * 3, 180 + label_h), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 8), label[:80], fill="black", font=font)
    for column, image in enumerate((before, after, diff)):
        tile.paste(image, (column * tile_w + (tile_w - image.width) // 2, label_h))
    return tile


def main() -> int:
    args = parse_args()
    pairs = read_pairs(args)
    if not pairs:
        raise ValueError("No before/after pairs found")

    font = ImageFont.load_default()
    tiles = [make_tile(before, after, label) for before, after, label in pairs]
    cols = 1 if len(tiles) == 1 else 2
    rows = (len(tiles) + cols - 1) // cols
    title_h = 34
    sheet = Image.new("RGB", (cols * tiles[0].width, rows * tiles[0].height + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), args.title, fill="black", font=font)
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile.width
        y = title_h + (index // cols) * tile.height
        sheet.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, "PNG")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
