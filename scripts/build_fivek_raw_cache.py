#!/usr/bin/env python3
"""Build a RAW-derived FiveK cache from the local FiveK tar and Expert TIFFs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import inspect_input, load_working_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FiveK RAW-derived auto-base cache.")
    parser.add_argument("--fivek-tar", type=Path, default=ROOT / "data" / "raw" / "fivek" / "fivek_dataset.tar")
    parser.add_argument("--expert-dir", type=Path, default=ROOT / "data" / "raw" / "fivek" / "expert_tiff" / "c")
    parser.add_argument("--expert", default="c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fivek_auto_optimize" / "raw_cache_v2_smoke")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-sheet-count", type=int, default=16)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def list_expert_targets(expert_dir: Path, count: int, seed: int) -> list[Path]:
    targets = sorted(expert_dir.glob("*.tif"))
    if not targets:
        raise FileNotFoundError(f"No Expert TIFFs found in {expert_dir}")
    rng = random.Random(seed)
    rng.shuffle(targets)
    selected = targets[:count] if count > 0 else targets
    return sorted(selected)


def build_tar_index(tar_path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name.lower().endswith((".dng", ".cr2", ".nef", ".arw")):
                index[Path(name).stem] = member.name
    return index


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


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def raw_render_image(raw_path: Path, size: int) -> tuple[Image.Image, dict]:
    working = load_working_image(raw_path)
    display = linear_to_srgb(working.pixels)
    image = Image.fromarray(np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas, {
        "working_space": working.working_space,
        "transfer_state": working.transfer_state,
        "warnings": [warning.__dict__ for warning in working.warnings],
    }


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


def save_pair(raw_image: Image.Image, target: Image.Image, path: Path) -> None:
    gutter = 8
    pair = Image.new("RGB", (raw_image.width + target.width + gutter, raw_image.height), "white")
    pair.paste(raw_image, (0, 0))
    pair.paste(target, (raw_image.width + gutter, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pair.save(path, "JPEG", quality=92, subsampling=1)


def make_contact_sheet(rows: list[dict[str, object]], output: Path, max_rows: int) -> None:
    font = ImageFont.load_default()
    sample = rows[:max_rows]
    tile_w = 180
    tile_h = 135
    row_h = tile_h + 34
    width = tile_w * 2 + 24
    height = 36 + len(sample) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "FiveK RAW cache V2 smoke: RAW/default | Expert C", fill="black", font=font)
    draw.text((8, 24), "RAW/default", fill="black", font=font)
    draw.text((tile_w + 16, 24), "Expert C", fill="black", font=font)
    for index, row in enumerate(sample):
        y = 36 + index * row_h
        raw_img = Image.open(ROOT / str(row["raw_render"])).convert("RGB")
        target = Image.open(ROOT / str(row["target"])).convert("RGB")
        for image in (raw_img, target):
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(raw_img, ((tile_w - raw_img.width) // 2, y))
        sheet.paste(target, (tile_w + 16 + (tile_w - target.width) // 2, y))
        draw.text((8, y + tile_h + 4), str(row["id"]), fill=(20, 20, 20), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / f"raw_renders_{args.size}"
    target_dir = output_dir / f"targets_{args.size}"
    pair_dir = output_dir / f"pairs_{args.size}"
    targets = list_expert_targets(args.expert_dir.resolve(), args.count, args.seed)
    tar_index = build_tar_index(args.fivek_tar.resolve())
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    with tarfile.open(args.fivek_tar.resolve(), "r") as tar, tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, target_path in enumerate(targets, start=1):
            stem = target_path.stem
            member_name = tar_index.get(stem)
            if member_name is None:
                missing.append(stem)
                continue
            member = tar.getmember(member_name)
            extracted = tar.extractfile(member)
            if extracted is None:
                missing.append(stem)
                continue
            raw_path = tmp_dir / Path(member_name).name
            raw_path.write_bytes(extracted.read())
            sample_id = f"{index:04d}_{stem}"
            raw_image, raw_decode_info = raw_render_image(raw_path, args.size)
            target = load_target(target_path, args.size)
            raw_out = raw_dir / f"{sample_id}_raw_default.jpg"
            target_out = target_dir / f"{sample_id}_expert_{args.expert}.png"
            pair_out = pair_dir / f"{sample_id}_pair.jpg"
            raw_out.parent.mkdir(parents=True, exist_ok=True)
            target_out.parent.mkdir(parents=True, exist_ok=True)
            raw_image.save(raw_out, "JPEG", quality=92, subsampling=1)
            target.save(target_out, "PNG")
            save_pair(raw_image, target, pair_out)
            raw_stats = image_stats(raw_image)
            target_stats = image_stats(target)
            inspection = inspect_input(raw_path)
            rows.append(
                {
                    "id": sample_id,
                    "source_name": stem,
                    "raw_tar_member": member_name,
                    "target_expert": args.expert,
                    "target_tiff": repo_path(target_path),
                    "raw_render": repo_path(raw_out),
                    "target": repo_path(target_out),
                    "pair": repo_path(pair_out),
                    "width": args.size,
                    "height": args.size,
                    "raw_working_space": raw_decode_info["working_space"],
                    "raw_transfer_state": raw_decode_info["transfer_state"],
                    "raw_warning_count": len(raw_decode_info["warnings"]),
                    "raw_visible_width": inspection.raw_metadata.get("visible_width", 0),
                    "raw_visible_height": inspection.raw_metadata.get("visible_height", 0),
                    "raw_luma_mean": raw_stats["luma_mean"],
                    "raw_luma_std": raw_stats["luma_std"],
                    "raw_chroma_mean": raw_stats["chroma_mean"],
                    "target_luma_mean": target_stats["luma_mean"],
                    "target_luma_std": target_stats["luma_std"],
                    "target_chroma_mean": target_stats["chroma_mean"],
                }
            )
            print(f"processed {len(rows)}/{len(targets)} {stem}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fivek_tar": repo_path(args.fivek_tar),
        "expert_dir": repo_path(args.expert_dir),
        "output_dir": repo_path(output_dir),
        "requested_count": args.count,
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing": missing,
        "size": args.size,
        "note": (
            "RAW/default render is generic LibRaw/rawpy output, not an exact vendor or Adobe rendering. "
            "Expert TIFF targets are converted from embedded ICC profiles to sRGB before caching."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(rows, output_dir / "contact_sheet.png", args.contact_sheet_count)
    print(f"rows={len(rows)} missing={len(missing)}")
    print(repo_path(output_dir))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
