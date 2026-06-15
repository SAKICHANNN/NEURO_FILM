#!/usr/bin/env python3
"""Build a bounded high-precision FiveK freeze pack before deleting raw data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import inspect_input, load_working_image  # noqa: E402


PROPHOTO_TO_XYZ_D50 = np.array(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0, 0.0, 0.82521],
    ],
    dtype=np.float32,
)
D50_TO_D65_BRADFORD = np.array(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float32,
)
XYZ_D65_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bounded FiveK freeze pack.")
    parser.add_argument("--fivek-tar", type=Path, default=ROOT / "data" / "raw" / "fivek" / "fivek_dataset.tar")
    parser.add_argument("--expert-dir", type=Path, default=ROOT / "data" / "raw" / "fivek" / "expert_tiff" / "c")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "fivek_auto_optimize" / "freeze_v1")
    parser.add_argument("--hp-count", type=int, default=128)
    parser.add_argument("--gold-count", type=int, default=64)
    parser.add_argument("--max-size", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--luma-strength", type=float, default=0.82)
    parser.add_argument("--chroma-strength", type=float, default=0.18)
    parser.add_argument("--chroma-headroom", type=float, default=0.02)
    parser.add_argument("--wb-anchor-strength", type=float, default=1.0)
    parser.add_argument("--max-gb", type=float, default=30.0)
    parser.add_argument("--contact-sheet-count", type=int, default=32)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def list_expert_targets(expert_dir: Path, count: int, seed: int) -> list[Path]:
    targets = sorted(expert_dir.glob("*.tif"))
    if not targets:
        raise FileNotFoundError(f"No Expert TIFF files found in {expert_dir}")
    rng = random.Random(seed)
    rng.shuffle(targets)
    return sorted(targets[:count])


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


def resize_float(rgb: np.ndarray, max_size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_size) / float(max(height, width)))
    if scale >= 1.0:
        return rgb.astype(np.float32, copy=False)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb.astype(np.float32, copy=False), size, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)


def resize_to_shape(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb.astype(np.float32, copy=False)
    return cv2.resize(rgb.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def prophoto_decode(encoded: np.ndarray) -> np.ndarray:
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(encoded < (16.0 / 512.0), encoded / 16.0, np.power(encoded, 1.8))


def load_expert_icc_srgb(path: Path, max_size: int) -> np.ndarray:
    data = tifffile.imread(path)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError(f"Unexpected Expert TIFF shape for {path}: {data.shape}")
    encoded = data[..., :3].astype(np.float32) / np.float32(np.iinfo(data.dtype).max)
    prophoto_linear = prophoto_decode(encoded)
    xyz_d50 = prophoto_linear @ PROPHOTO_TO_XYZ_D50.T
    xyz_d65 = xyz_d50 @ D50_TO_D65_BRADFORD.T
    srgb_linear = xyz_d65 @ XYZ_D65_TO_SRGB.T
    return resize_float(srgb_encode(srgb_linear), max_size)


def load_raw_default(raw_path: Path, max_size: int) -> tuple[np.ndarray, dict]:
    working = load_working_image(raw_path)
    srgb = srgb_encode(working.pixels)
    return resize_float(srgb, max_size), {
        "working_space": working.working_space,
        "transfer_state": working.transfer_state,
        "warnings": [warning.__dict__ for warning in working.warnings],
    }


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def chroma_vector(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lum = luma(rgb)
    return lum, rgb - lum[..., None]


def chroma(rgb: np.ndarray) -> np.ndarray:
    _, vec = chroma_vector(rgb)
    return np.sqrt((vec * vec).sum(axis=2))


def robust_channel_mean(rgb: np.ndarray) -> np.ndarray:
    lum = luma(rgb)
    mask = (lum > 0.05) & (lum < 0.95)
    if not np.any(mask):
        return rgb.reshape(-1, 3).mean(axis=0)
    return rgb[mask].reshape(-1, 3).mean(axis=0)


def anchor_white_balance(raw: np.ndarray, candidate: np.ndarray, strength: float) -> np.ndarray:
    raw_mean = robust_channel_mean(raw)
    candidate_mean = robust_channel_mean(candidate)
    raw_rg = raw_mean[0] / max(raw_mean[1], 1e-6)
    raw_bg = raw_mean[2] / max(raw_mean[1], 1e-6)
    candidate_rg = candidate_mean[0] / max(candidate_mean[1], 1e-6)
    candidate_bg = candidate_mean[2] / max(candidate_mean[1], 1e-6)
    gains = np.array(
        [
            (raw_rg / max(candidate_rg, 1e-6)) ** float(strength),
            1.0,
            (raw_bg / max(candidate_bg, 1e-6)) ** float(strength),
        ],
        dtype=np.float32,
    )
    adjusted = np.clip(candidate * gains[None, None, :], 0.0, 1.0)
    candidate_l = luma(candidate)
    adjusted_l = luma(adjusted)
    return np.clip(adjusted * (candidate_l / np.maximum(adjusted_l, 1e-4))[..., None], 0.0, 1.0)


def filtered_target(raw: np.ndarray, expert: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    raw_l, raw_vec = chroma_vector(raw)
    expert_l, expert_vec = chroma_vector(expert)
    target_l = raw_l + (expert_l - raw_l) * float(args.luma_strength)
    target_vec = raw_vec + (expert_vec - raw_vec) * float(args.chroma_strength)
    raw_c = np.sqrt((raw_vec * raw_vec).sum(axis=2))
    target_c = np.sqrt((target_vec * target_vec).sum(axis=2))
    max_c = raw_c * (1.0 + float(args.chroma_headroom))
    scale = np.minimum(1.0, max_c / np.maximum(target_c, 1e-6))
    filtered = np.clip(target_l[..., None] + target_vec * scale[..., None], 0.0, 1.0)
    return anchor_white_balance(raw, filtered, args.wb_anchor_strength)


def save_tiff16(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
    tifffile.imwrite(path, data, photometric="rgb", metadata={"axes": "YXS"})


def save_preview(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "JPEG", quality=92)


def image_stats(raw: np.ndarray, expert: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
    raw_l = luma(raw)
    expert_l = luma(expert)
    filtered_l = luma(filtered)
    raw_c = chroma(raw)
    expert_c = chroma(expert)
    filtered_c = chroma(filtered)
    return {
        "expert_raw_rgb_mae": float(np.abs(expert - raw).mean()),
        "filtered_raw_rgb_mae": float(np.abs(filtered - raw).mean()),
        "expert_luma_delta_mean": float((expert_l - raw_l).mean()),
        "filtered_luma_delta_mean": float((filtered_l - raw_l).mean()),
        "expert_chroma_delta_mean": float((expert_c - raw_c).mean()),
        "filtered_chroma_delta_mean": float((filtered_c - raw_c).mean()),
    }


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def make_contact_sheet(rows: list[dict[str, object]], output: Path, max_rows: int) -> None:
    font = ImageFont.load_default()
    sample = rows[:max_rows]
    tile_w = 180
    tile_h = 135
    row_h = tile_h + 34
    width = tile_w * 4 + 40
    height = 38 + len(sample) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "FiveK freeze v1: RAW/default | filtered target | Expert C ICC | diff x4", fill="black", font=font)
    labels = ["RAW/default", "filtered", "Expert C ICC", "diff x4"]
    for col, label in enumerate(labels):
        draw.text((8 + col * (tile_w + 8), 24), label, fill="black", font=font)
    for index, row in enumerate(sample):
        y = 38 + index * row_h
        raw = Image.open(ROOT / str(row["raw_preview"])).convert("RGB")
        filtered = Image.open(ROOT / str(row["filtered_preview"])).convert("RGB")
        expert = Image.open(ROOT / str(row["expert_preview"])).convert("RGB")
        raw_arr = np.asarray(raw, dtype=np.float32) / 255.0
        filtered_arr = np.asarray(filtered, dtype=np.float32) / 255.0
        diff = Image.fromarray(np.rint(np.clip(np.abs(filtered_arr - raw_arr) * 4.0, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
        images = [raw, filtered, expert, diff]
        for image in images:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        for col, image in enumerate(images):
            x = 8 + col * (tile_w + 8)
            sheet.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.text((8, y + tile_h + 4), str(row["id"]), fill=(20, 20, 20), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or ROOT not in output_dir.parents:
        raise ValueError(f"Refusing unexpected output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = list_expert_targets(args.expert_dir.resolve(), args.hp_count, args.seed)
    tar_index = build_tar_index(args.fivek_tar.resolve())
    hp_dir = output_dir / f"hp{args.hp_count}_{args.max_size}_srgb16"
    gold_dir = output_dir / f"gold{args.gold_count}_original_samples"
    tmp_extract_dir = output_dir / "_tmp_extract"
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    with tarfile.open(args.fivek_tar.resolve(), "r") as tar, tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        tmp_dir = Path(tmp)
        for index, expert_path in enumerate(targets, start=1):
            stem = expert_path.stem
            member_name = tar_index.get(stem)
            if member_name is None:
                missing.append(stem)
                continue
            extracted = tar.extractfile(tar.getmember(member_name))
            if extracted is None:
                missing.append(stem)
                continue
            raw_tmp = tmp_dir / Path(member_name).name
            raw_tmp.write_bytes(extracted.read())
            sample_id = f"{index:04d}_{stem}"
            raw_rgb, raw_info = load_raw_default(raw_tmp, args.max_size)
            expert_rgb = resize_to_shape(load_expert_icc_srgb(expert_path, args.max_size), raw_rgb.shape[:2])
            filtered_rgb = filtered_target(raw_rgb, expert_rgb, args)

            raw_tiff = hp_dir / "raw_default_srgb16" / f"{sample_id}_raw_default_srgb16.tif"
            expert_tiff = hp_dir / "expert_c_icc_srgb16" / f"{sample_id}_expert_c_icc_srgb16.tif"
            filtered_tiff = hp_dir / "filtered_target_srgb16" / f"{sample_id}_filtered_target_srgb16.tif"
            raw_preview = hp_dir / "previews" / "raw_default" / f"{sample_id}_raw_default.jpg"
            expert_preview = hp_dir / "previews" / "expert_c_icc" / f"{sample_id}_expert_c_icc.jpg"
            filtered_preview = hp_dir / "previews" / "filtered_target" / f"{sample_id}_filtered_target.jpg"

            save_tiff16(raw_rgb, raw_tiff)
            save_tiff16(expert_rgb, expert_tiff)
            save_tiff16(filtered_rgb, filtered_tiff)
            save_preview(raw_rgb, raw_preview)
            save_preview(expert_rgb, expert_preview)
            save_preview(filtered_rgb, filtered_preview)

            raw_gold = ""
            expert_gold = ""
            if index <= args.gold_count:
                raw_gold_path = gold_dir / "raw" / Path(member_name).name
                expert_gold_path = gold_dir / "expert_tiff_c" / expert_path.name
                raw_gold_path.parent.mkdir(parents=True, exist_ok=True)
                expert_gold_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_tmp, raw_gold_path)
                shutil.copy2(expert_path, expert_gold_path)
                raw_gold = repo_path(raw_gold_path)
                expert_gold = repo_path(expert_gold_path)

            inspection = inspect_input(raw_tmp)
            stats = image_stats(raw_rgb, expert_rgb, filtered_rgb)
            row: dict[str, object] = {
                "id": sample_id,
                "source_name": stem,
                "raw_tar_member": member_name,
                "expert_tiff_source": repo_path(expert_path),
                "raw_default_srgb16": repo_path(raw_tiff),
                "expert_c_icc_srgb16": repo_path(expert_tiff),
                "filtered_target_srgb16": repo_path(filtered_tiff),
                "raw_preview": repo_path(raw_preview),
                "expert_preview": repo_path(expert_preview),
                "filtered_preview": repo_path(filtered_preview),
                "raw_gold": raw_gold,
                "expert_gold": expert_gold,
                "width": raw_rgb.shape[1],
                "height": raw_rgb.shape[0],
                "raw_working_space": raw_info["working_space"],
                "raw_transfer_state": raw_info["transfer_state"],
                "raw_warning_count": len(raw_info["warnings"]),
                "raw_visible_width": inspection.raw_metadata.get("visible_width", 0),
                "raw_visible_height": inspection.raw_metadata.get("visible_height", 0),
                **stats,
            }
            rows.append(row)
            if index % 8 == 0 or index == len(targets):
                current_gb = dir_size(output_dir) / 1_073_741_824
                print(f"processed {index}/{len(targets)} freeze_size_gb={current_gb:.2f}", flush=True)
                if current_gb > args.max_gb:
                    raise RuntimeError(f"Freeze pack exceeded max budget {args.max_gb} GB")
            raw_tmp.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_path(output_dir),
        "hp_count": args.hp_count,
        "gold_count": args.gold_count,
        "max_size": args.max_size,
        "max_gb": args.max_gb,
        "actual_gb": dir_size(output_dir) / 1_073_741_824,
        "missing": missing,
        "filter_params": {
            "luma_strength": args.luma_strength,
            "chroma_strength": args.chroma_strength,
            "chroma_headroom": args.chroma_headroom,
            "wb_anchor_strength": args.wb_anchor_strength,
        },
        "expert_color_management": "16-bit TIFF read via tifffile; ProPhoto/ROMM transfer decoded; Bradford D50->D65; matrix to sRGB; sRGB transfer encoded to uint16 TIFF.",
        "raw_color_management": "rawpy/LibRaw generic camera decode via src.preprocess; display sRGB transfer encoded to uint16 TIFF.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(rows, output_dir / "contact_sheet.png", args.contact_sheet_count)
    print(f"rows={len(rows)} missing={len(missing)}")
    print(f"actual_gb={summary['actual_gb']:.2f}")
    print(repo_path(output_dir))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
