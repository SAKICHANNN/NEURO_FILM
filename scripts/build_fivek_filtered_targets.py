#!/usr/bin/env python3
"""Build conservative FiveK filtered targets from ICC-corrected Expert C cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative FiveK filtered targets.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "raw_cache_v2_mini64_icc" / "manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "filtered_targets_v1_mini64_icc",
    )
    parser.add_argument("--luma-strength", type=float, default=0.82)
    parser.add_argument("--chroma-strength", type=float, default=0.18)
    parser.add_argument("--chroma-headroom", type=float, default=0.02)
    parser.add_argument("--wb-anchor-strength", type=float, default=1.0)
    parser.add_argument("--contact-sheet-count", type=int, default=24)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


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


def build_filtered_target(
    raw: np.ndarray,
    expert: np.ndarray,
    luma_strength: float,
    chroma_strength: float,
    chroma_headroom: float,
    wb_anchor_strength: float,
) -> np.ndarray:
    raw_l, raw_vec = chroma_vector(raw)
    expert_l, expert_vec = chroma_vector(expert)
    target_l = raw_l + (expert_l - raw_l) * float(luma_strength)
    target_vec = raw_vec + (expert_vec - raw_vec) * float(chroma_strength)
    raw_c = np.sqrt((raw_vec * raw_vec).sum(axis=2))
    target_c = np.sqrt((target_vec * target_vec).sum(axis=2))
    max_c = raw_c * (1.0 + float(chroma_headroom))
    scale = np.minimum(1.0, max_c / np.maximum(target_c, 1e-6))
    filtered = np.clip(target_l[..., None] + target_vec * scale[..., None], 0.0, 1.0)
    return anchor_white_balance(raw, filtered, wb_anchor_strength)


def image_metrics(raw: np.ndarray, expert: np.ndarray, filtered: np.ndarray) -> dict[str, float]:
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
        "filtered_expert_rgb_mae": float(np.abs(filtered - expert).mean()),
    }


def make_contact_sheet(rows: list[dict[str, str]], output: Path, max_rows: int) -> None:
    font = ImageFont.load_default()
    sample = rows[:max_rows]
    tile_w = 160
    tile_h = 120
    row_h = tile_h + 32
    width = tile_w * 4 + 40
    height = 38 + len(sample) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "FiveK filtered targets: RAW | filtered target | Expert C ICC | diff x4", fill="black", font=font)
    labels = [("RAW", 8), ("filtered", tile_w + 16), ("Expert C", tile_w * 2 + 24), ("diff x4", tile_w * 3 + 32)]
    for label, x in labels:
        draw.text((x, 24), label, fill="black", font=font)
    for index, row in enumerate(sample):
        y = 38 + index * row_h
        raw = Image.open(ROOT / row["raw_render"]).convert("RGB")
        filtered = Image.open(ROOT / row["target"]).convert("RGB")
        expert = Image.open(ROOT / row["expert_target"]).convert("RGB")
        raw_arr = np.asarray(raw, dtype=np.float32) / 255.0
        filtered_arr = np.asarray(filtered, dtype=np.float32) / 255.0
        diff = Image.fromarray(np.rint(np.clip(np.abs(filtered_arr - raw_arr) * 4.0, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
        images = [raw, filtered, expert, diff]
        for image in images:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        xs = [8, tile_w + 16, tile_w * 2 + 24, tile_w * 3 + 32]
        for x, image in zip(xs, images):
            sheet.paste(image, (x + (tile_w - image.width) // 2, y))
        draw.text((8, y + tile_h + 4), row["id"], fill=(20, 20, 20), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": repo_path(args.manifest),
        "output_dir": repo_path(args.output_dir),
        "luma_strength": args.luma_strength,
        "chroma_strength": args.chroma_strength,
        "chroma_headroom": args.chroma_headroom,
        "wb_anchor_strength": args.wb_anchor_strength,
        "row_count": len(rows),
        "means": {
            "expert_raw_rgb_mae": mean("expert_raw_rgb_mae"),
            "filtered_raw_rgb_mae": mean("filtered_raw_rgb_mae"),
            "expert_luma_delta_mean": mean("expert_luma_delta_mean"),
            "filtered_luma_delta_mean": mean("filtered_luma_delta_mean"),
            "expert_chroma_delta_mean": mean("expert_chroma_delta_mean"),
            "filtered_chroma_delta_mean": mean("filtered_chroma_delta_mean"),
            "filtered_expert_rgb_mae": mean("filtered_expert_rgb_mae"),
        },
        "note": "Filtered targets keep much of Expert C luma response while guarding WB and chroma growth.",
    }


def main() -> int:
    args = parse_args()
    source_rows = read_manifest(args.manifest)
    if not source_rows:
        raise ValueError(f"No rows in manifest: {args.manifest}")
    output_dir = args.output_dir.resolve()
    target_dir = output_dir / "targets"
    out_rows: list[dict[str, object]] = []
    sheet_rows: list[dict[str, str]] = []
    for row in source_rows:
        raw = load_rgb(ROOT / row["raw_render"])
        expert = load_rgb(ROOT / row["target"])
        filtered = build_filtered_target(
            raw,
            expert,
            args.luma_strength,
            args.chroma_strength,
            args.chroma_headroom,
            args.wb_anchor_strength,
        )
        target_path = target_dir / f"{row['id']}_filtered_target.png"
        save_rgb(filtered, target_path)
        metrics = image_metrics(raw, expert, filtered)
        out_row: dict[str, object] = {
            **row,
            "expert_target": row["target"],
            "target": repo_path(target_path),
            "filter_luma_strength": args.luma_strength,
            "filter_chroma_strength": args.chroma_strength,
            "filter_chroma_headroom": args.chroma_headroom,
            "filter_wb_anchor_strength": args.wb_anchor_strength,
            **metrics,
        }
        out_rows.append(out_row)
        sheet_rows.append({key: str(out_row[key]) for key in ("id", "raw_render", "target", "expert_target")})

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (output_dir / "summary.json").write_text(json.dumps(summarize(out_rows, args), indent=2), encoding="utf-8")
    make_contact_sheet(sheet_rows, output_dir / "contact_sheet.png", args.contact_sheet_count)
    print(f"rows={len(out_rows)}")
    print(repo_path(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
