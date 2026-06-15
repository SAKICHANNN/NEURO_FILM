#!/usr/bin/env python3
"""Evaluate a deterministic FiveK response baseline from compact stats."""

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
    parser = argparse.ArgumentParser(description="Evaluate FiveK deterministic response baseline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "raw_cache_v2_mini64" / "manifest.csv",
    )
    parser.add_argument(
        "--response",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "response_stats_v1_mini64" / "response_curves.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "fivek_auto_optimize" / "response_baseline_v1_mini64",
    )
    parser.add_argument("--contact-sheet-count", type=int, default=24)
    parser.add_argument("--strength", type=float, default=1.0)
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


def chroma(rgb: np.ndarray) -> np.ndarray:
    return np.sqrt(((rgb - rgb.mean(axis=2, keepdims=True)) ** 2).sum(axis=2))


def apply_response(raw: np.ndarray, curves: np.lib.npyio.NpzFile, strength: float) -> np.ndarray:
    raw_l = luma(raw)
    centers = curves["bin_centers"]
    rgb_delta = curves["rgb_delta_by_raw_luma"]
    delta_r = np.interp(raw_l, centers, rgb_delta[:, 0])
    delta_g = np.interp(raw_l, centers, rgb_delta[:, 1])
    delta_b = np.interp(raw_l, centers, rgb_delta[:, 2])
    delta = np.stack([delta_r, delta_g, delta_b], axis=2).astype(np.float32)
    return np.clip(raw + delta * float(strength), 0.0, 1.0)


def metrics(raw: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict[str, float]:
    raw_l = luma(raw)
    base_l = luma(baseline)
    target_l = luma(target)
    raw_c = chroma(raw)
    base_c = chroma(baseline)
    target_c = chroma(target)
    return {
        "raw_target_luma_mae": float(np.abs(raw_l - target_l).mean()),
        "baseline_target_luma_mae": float(np.abs(base_l - target_l).mean()),
        "raw_target_rgb_mae": float(np.abs(raw - target).mean()),
        "baseline_target_rgb_mae": float(np.abs(baseline - target).mean()),
        "raw_target_chroma_mae": float(np.abs(raw_c - target_c).mean()),
        "baseline_target_chroma_mae": float(np.abs(base_c - target_c).mean()),
        "baseline_luma_delta_mean": float((base_l - raw_l).mean()),
        "baseline_chroma_delta_mean": float((base_c - raw_c).mean()),
    }


def make_contact_sheet(rows: list[dict[str, str]], output: Path, max_rows: int) -> None:
    font = ImageFont.load_default()
    sample = rows[:max_rows]
    tile_w = 160
    tile_h = 120
    row_h = tile_h + 32
    width = tile_w * 3 + 32
    height = 38 + len(sample) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "FiveK response baseline: RAW/default | baseline | Expert C", fill="black", font=font)
    for x, label in [(8, "RAW/default"), (tile_w + 16, "baseline"), (tile_w * 2 + 24, "Expert C")]:
        draw.text((x, 24), label, fill="black", font=font)
    for index, row in enumerate(sample):
        y = 38 + index * row_h
        images = [
            Image.open(ROOT / row["raw_render"]).convert("RGB"),
            Image.open(ROOT / row["baseline"]).convert("RGB"),
            Image.open(ROOT / row["target"]).convert("RGB"),
        ]
        for image in images:
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        xs = [8, tile_w + 16, tile_w * 2 + 24]
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
        "response": repo_path(args.response),
        "output_dir": repo_path(args.output_dir),
        "row_count": len(rows),
        "strength": args.strength,
        "means": {
            "raw_target_luma_mae": mean("raw_target_luma_mae"),
            "baseline_target_luma_mae": mean("baseline_target_luma_mae"),
            "raw_target_rgb_mae": mean("raw_target_rgb_mae"),
            "baseline_target_rgb_mae": mean("baseline_target_rgb_mae"),
            "raw_target_chroma_mae": mean("raw_target_chroma_mae"),
            "baseline_target_chroma_mae": mean("baseline_target_chroma_mae"),
            "baseline_luma_delta_mean": mean("baseline_luma_delta_mean"),
            "baseline_chroma_delta_mean": mean("baseline_chroma_delta_mean"),
        },
        "note": "Deterministic RGB-delta response baseline from compact stats; not a trained final auto-base model.",
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    baseline_dir = output_dir / "baseline"
    rows = read_manifest(args.manifest)
    curves = np.load(args.response)
    out_rows: list[dict[str, object]] = []
    sheet_rows: list[dict[str, str]] = []
    for row in rows:
        raw = load_rgb(ROOT / row["raw_render"])
        target = load_rgb(ROOT / row["target"])
        baseline = apply_response(raw, curves, args.strength)
        baseline_path = baseline_dir / f"{row['id']}_response_baseline.png"
        save_rgb(baseline, baseline_path)
        metric = metrics(raw, baseline, target)
        out_row: dict[str, object] = {
            "id": row["id"],
            "raw_render": row["raw_render"],
            "baseline": repo_path(baseline_path),
            "target": row["target"],
            **metric,
        }
        out_rows.append(out_row)
        sheet_rows.append({key: str(out_row[key]) for key in ("id", "raw_render", "baseline", "target")})

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
