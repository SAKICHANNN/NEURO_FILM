#!/usr/bin/env python3
"""Build a compact FiveK auto-optimization paired cache."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageEnhance, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact FiveK auto-optimize cache.")
    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "fivek" / "expert_tiff" / "c",
    )
    parser.add_argument("--expert", default="c")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "cache_v1",
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-sheet-count", type=int, default=24)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def list_targets(expert_dir: Path, count: int, seed: int) -> list[Path]:
    if not expert_dir.exists():
        raise FileNotFoundError(f"Missing FiveK expert directory: {expert_dir}")
    targets = sorted(path for path in expert_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not targets:
        raise FileNotFoundError(f"No target images found in {expert_dir}")
    rng = random.Random(seed)
    rng.shuffle(targets)
    if count > 0:
        targets = targets[:count]
    return sorted(targets)


def convert_to_srgb(image: Image.Image) -> Image.Image:
    icc = image.info.get("icc_profile")
    if not icc:
        return image.convert("RGB")
    src = ImageCms.ImageCmsProfile(BytesIO(icc))
    dst = ImageCms.createProfile("sRGB")
    return ImageCms.profileToProfile(image.convert("RGB"), src, dst, outputMode="RGB")


def load_target(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = convert_to_srgb(ImageOps.exif_transpose(image))
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "black")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_input_proxy(target: Image.Image) -> Image.Image:
    """Create a conservative unoptimized-display proxy from an Expert TIFF target."""

    image = ImageEnhance.Color(target).enhance(0.72)
    image = ImageEnhance.Contrast(image).enhance(0.84)
    image = ImageEnhance.Brightness(image).enhance(0.96)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Slightly lift shadows and compress upper midtones to mimic a flatter base render.
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, 1.08)
    arr = np.clip(arr * 0.96 + 0.018, 0.0, 1.0)
    return Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB")


def image_stats(image: Image.Image) -> dict[str, float | int]:
    arr = np.asarray(image, dtype=np.float32)
    rgb = arr / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    chroma = np.sqrt(((rgb - rgb.mean(axis=2, keepdims=True)) ** 2).sum(axis=2))
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "chroma_mean": float(chroma.mean()),
    }


def save_pair(input_proxy: Image.Image, target: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    gutter = 8
    width = input_proxy.width + target.width + gutter
    height = input_proxy.height
    pair = Image.new("RGB", (width, height), "white")
    pair.paste(input_proxy, (0, 0))
    pair.paste(target, (input_proxy.width + gutter, 0))
    pair.save(output, "JPEG", quality=92, subsampling=1)


def make_contact_sheet(rows: list[dict[str, object]], output: Path, max_rows: int) -> None:
    font = ImageFont.load_default()
    sample = rows[:max_rows]
    tile_w = 180
    tile_h = 135
    row_h = tile_h + 34
    columns = 2
    width = columns * tile_w + 24
    height = 36 + len(sample) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "FiveK auto optimize cache v1: input proxy | Expert C target", fill="black", font=font)
    draw.text((8, 24), "input proxy", fill="black", font=font)
    draw.text((tile_w + 16, 24), "target", fill="black", font=font)

    for index, row in enumerate(sample):
        y = 36 + index * row_h
        inp = Image.open(ROOT / str(row["input_proxy"])).convert("RGB")
        tgt = Image.open(ROOT / str(row["target_512"])).convert("RGB")
        for image in (inp, tgt):
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(inp, ((tile_w - inp.width) // 2, y))
        sheet.paste(tgt, (tile_w + 16 + (tile_w - tgt.width) // 2, y))
        draw.text((8, y + tile_h + 4), str(row["id"]), fill=(20, 20, 20), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "MIT-Adobe FiveK Expert TIFF cache",
        "expert": args.expert,
        "expert_dir": repo_path(args.expert_dir),
        "output_dir": repo_path(args.output_dir),
        "count": len(rows),
        "size": args.size,
        "input_proxy_policy": {
            "color_scale": 0.72,
            "contrast_scale": 0.84,
            "brightness_scale": 0.96,
            "gamma": 1.08,
            "post_scale": 0.96,
            "post_offset": 0.018,
            "note": "Proxy only; true RAW-derived inputs are a future cache version.",
        },
        "target_color_management": "embedded ICC converted to sRGB before cache generation",
        "means": {
            "target_luma_mean": mean("target_luma_mean"),
            "target_luma_std": mean("target_luma_std"),
            "target_chroma_mean": mean("target_chroma_mean"),
            "input_luma_mean": mean("input_luma_mean"),
            "input_luma_std": mean("input_luma_std"),
            "input_chroma_mean": mean("input_chroma_mean"),
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    input_dir = output_dir / f"inputs_{args.size}"
    target_dir = output_dir / f"targets_{args.size}"
    pair_dir = output_dir / f"pairs_{args.size}"
    targets = list_targets(args.expert_dir.resolve(), args.count, args.seed)
    rows: list[dict[str, object]] = []

    for index, path in enumerate(targets, start=1):
        sample_id = f"{index:04d}_{path.stem}"
        target = load_target(path, args.size)
        input_proxy = make_input_proxy(target)
        input_path = input_dir / f"{sample_id}_input_proxy.jpg"
        target_path = target_dir / f"{sample_id}_expert_{args.expert}.png"
        pair_path = pair_dir / f"{sample_id}_pair.jpg"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        input_proxy.save(input_path, "JPEG", quality=92, subsampling=1)
        target.save(target_path, "PNG")
        save_pair(input_proxy, target, pair_path)

        input_stats = image_stats(input_proxy)
        target_stats = image_stats(target)
        rows.append(
            {
                "id": sample_id,
                "source_name": path.name,
                "target_expert": args.expert,
                "target_tiff": repo_path(path),
                "input_proxy": repo_path(input_path),
                "target_512": repo_path(target_path),
                "pair_512": repo_path(pair_path),
                "width": args.size,
                "height": args.size,
                "input_min": input_stats["min"],
                "input_max": input_stats["max"],
                "input_luma_mean": input_stats["luma_mean"],
                "input_luma_std": input_stats["luma_std"],
                "input_chroma_mean": input_stats["chroma_mean"],
                "target_min": target_stats["min"],
                "target_max": target_stats["max"],
                "target_luma_mean": target_stats["luma_mean"],
                "target_luma_std": target_stats["luma_std"],
                "target_chroma_mean": target_stats["chroma_mean"],
            }
        )
        if index % 25 == 0 or index == len(targets):
            print(f"processed {index}/{len(targets)}", flush=True)

    manifest = output_dir / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summarize(rows, args), indent=2), encoding="utf-8")
    make_contact_sheet(rows, output_dir / "contact_sheet.png", args.contact_sheet_count)

    print(f"rows={len(rows)}")
    print(repo_path(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
