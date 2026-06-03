#!/usr/bin/env python3
"""Evaluate distilled SepLUT outputs for Neural Film LUT V2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from src.models.neural_film_lut.distilled_seplut import apply_distilled_seplut  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled film SepLUT.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=ROOT / "outputs" / "eval" / "color_engine_challenge" / "film_response_v1_s1p0",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "eval" / "neural_film_lut_v2")
    parser.add_argument("--run-prefix", default="scheme_a_distilled")
    parser.add_argument("--styles", default="")
    parser.add_argument("--strengths", default="s0p5:0.5,s1p0:1.0,s1p5:1.5,s2p0:2.0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--output-margin", type=int, default=4)
    parser.add_argument("--debug-progress", action="store_true")
    return parser.parse_args()


def parse_strengths(raw: str) -> list[tuple[str, float]]:
    pairs = []
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split(":", 1)
        pairs.append((label.strip(), float(value)))
    return pairs


def source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    return [Path(row.get("before") or row.get("input")) for row in rows if Path(row.get("before") or row.get("input")).exists()]


def load_rgb(path: Path, max_side: int = 0) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_rgb_u8(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def make_diff_map(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).astype(np.float32)
    diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff, mode="RGB").save(path, "PNG")


def fast_evaluate(before_path: Path, after_path: Path) -> dict:
    before = load_rgb_u8(before_path)
    after = load_rgb_u8(after_path)
    before_float = before.astype(np.float32) / 255.0
    after_float = after.astype(np.float32) / 255.0
    before_luma = before_float @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    after_luma = after_float @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_delta = np.abs(after_luma - before_luma)
    before_clip = np.any((before <= 0) | (before >= 255), axis=2)
    after_clip = np.any((after <= 0) | (after >= 255), axis=2)
    before_chroma = before_float.max(axis=2) - before_float.min(axis=2)
    after_chroma = after_float.max(axis=2) - after_float.min(axis=2)
    neutral = before_chroma < (8.0 / 255.0)
    contaminated = neutral & (after_chroma > (14.0 / 255.0))
    return {
        "clip": {
            "after_min": int(after.min()),
            "after_max": int(after.max()),
            "new_clipped_pixel_count": int((after_clip & ~before_clip).sum()),
            "new_clipped_pixel_percent": float((after_clip & ~before_clip).mean() * 100.0),
        },
        "structure": {
            "l_ssim": float(max(0.0, 1.0 - luma_delta.mean() * 6.0)),
            "gradient_delta_p95": float(np.percentile(luma_delta, 95)),
            "high_frequency_delta_p95": float(np.percentile(np.abs(luma_delta - luma_delta.mean()), 95)),
        },
        "color": {
            "after_chroma_mean": float(after_chroma.mean() * 100.0),
            "neutral_contaminated_percent": float(contaminated.mean() * 100.0),
        },
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "image_count": len(rows),
        "new_clip_image_count": sum(1 for row in rows if row["new_clipped_pixel_count"] > 0),
        "hard_bound_image_count": sum(1 for row in rows if row["after_min"] <= 0 or row["after_max"] >= 255),
        "new_clipped_pixel_count_total": int(sum(row["new_clipped_pixel_count"] for row in rows)),
        "l_ssim_min": min(row["l_ssim"] for row in rows),
        "l_ssim_mean": mean(row["l_ssim"] for row in rows),
        "gradient_delta_p95_mean": mean(row["gradient_delta_p95"] for row in rows),
        "high_frequency_delta_p95_mean": mean(row["high_frequency_delta_p95"] for row in rows),
        "after_chroma_mean": mean(row["after_chroma_mean"] for row in rows),
        "neutral_contaminated_percent_max": max(row["neutral_contaminated_percent"] for row in rows),
        "after_min_min": min(row["after_min"] for row in rows),
        "after_max_max": max(row["after_max"] for row in rows),
        "gate_mean": mean(row["gate"] for row in rows),
    }


def contact_sheet(rows: list[dict], path: Path, title: str) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for row in rows:
        before = Image.open(row["before"]).convert("RGB")
        safe = Image.open(row["safe"]).convert("RGB")
        after = Image.open(row["after"]).convert("RGB")
        for image in (before, safe, after):
            image.thumbnail((210, 158), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (630, 198), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"{row['id']} L={row['l_ssim']:.5f} C={row['after_chroma_mean']:.2f} G={row['gate']:.2f}", fill="black", font=font)
        tile.paste(before, ((210 - before.width) // 2, 34))
        tile.paste(safe, (210 + (210 - safe.width) // 2, 34))
        tile.paste(after, (420 + (210 - after.width) // 2, 34))
        draw.text((8, 178), "before", fill="black", font=font)
        draw.text((218, 178), "teacher-safe", fill="black", font=font)
        draw.text((428, 178), "distilled", fill="black", font=font)
        thumbs.append(tile)
    sheet = Image.new("RGB", (630, len(thumbs) * 198 + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black", font=font)
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, (0, 30 + index * 198))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")


def main() -> int:
    args = parse_args()
    if args.debug_progress:
        print("loading_model", flush=True)
    model = np.load(args.model, allow_pickle=False)
    styles = [str(item) for item in model["styles"].tolist()]
    if args.styles:
        requested = [item.strip() for item in args.styles.split(",") if item.strip()]
        styles = [style for style in styles if style in requested]
    strengths = parse_strengths(args.strengths)
    if args.debug_progress:
        print("loading_sources", flush=True)
    paths = source_paths(args.source_manifest, args.limit)
    output_root = args.output_root.resolve()
    style_to_index = {style: index for index, style in enumerate(model["styles"].tolist())}
    summaries = {}
    for label, strength in strengths:
        if args.debug_progress:
            print(f"run {label}", flush=True)
        run_name = f"{args.run_prefix}_{label}"
        run_dir = output_root / run_name
        summary = {
            "engine": "distilled_style_separated_seplut",
            "model": str(args.model),
            "image_count": len(paths),
            "strength": strength,
            "styles": {},
        }
        for style in styles:
            rows = []
            style_dir = run_dir / style
            style_index = style_to_index[style]
            for index, input_path in enumerate(paths, start=1):
                if args.debug_progress:
                    print(f"start {label} {style} {index}", flush=True)
                source = load_rgb(input_path, args.max_side)
                if args.debug_progress:
                    print("apply_lut", flush=True)
                out, gate = apply_distilled_seplut(
                    source,
                    style_index=style_index,
                    lut1d=model["lut1d"],
                    residual3d=model["residual3d"],
                    gate_weights=model["gate_weights"],
                    strength=strength,
                    output_margin=args.output_margin,
                )
                after_path = style_dir / "after" / f"{index:02d}_{style}_distilled.png"
                before_path = style_dir / "before" / f"{index:02d}_{style}_before.png"
                diff_path = style_dir / "diff_maps" / f"{index:02d}_{style}_diff.png"
                safe_src = args.safe_root / style / "safe_rich" / f"{index:02d}_{style}_safe_rich.png"
                safe_path = style_dir / "safe_rich" / f"{index:02d}_{style}_safe_rich.png"
                if args.debug_progress:
                    print("save_images", flush=True)
                save_rgb(source, before_path)
                save_rgb(out, after_path)
                if safe_src.exists() and not safe_path.exists():
                    save_rgb(load_rgb(safe_src, args.max_side), safe_path)
                elif not safe_path.exists():
                    save_rgb(source, safe_path)
                if args.debug_progress:
                    print("evaluate", flush=True)
                metrics = fast_evaluate(before_path, after_path)
                make_diff_map(load_rgb_u8(before_path), load_rgb_u8(after_path), diff_path)
                row = {
                    "id": f"{index:02d}",
                    "before": str(before_path),
                    "source_original": str(input_path),
                    "safe": str(safe_path),
                    "after": str(after_path),
                    "diff_map": str(diff_path),
                    "after_min": metrics["clip"]["after_min"],
                    "after_max": metrics["clip"]["after_max"],
                    "new_clipped_pixel_count": metrics["clip"]["new_clipped_pixel_count"],
                    "new_clipped_pixel_percent": metrics["clip"]["new_clipped_pixel_percent"],
                    "l_ssim": metrics["structure"]["l_ssim"],
                    "gradient_delta_p95": metrics["structure"]["gradient_delta_p95"],
                    "high_frequency_delta_p95": metrics["structure"]["high_frequency_delta_p95"],
                    "after_chroma_mean": metrics["color"]["after_chroma_mean"],
                    "neutral_contaminated_percent": metrics["color"]["neutral_contaminated_percent"],
                    "gate": gate,
                }
                rows.append(row)
                print(f"{label} {style} {index:02d} Lssim={row['l_ssim']:.5f} chroma={row['after_chroma_mean']:.2f} gate={gate:.2f}")
            style_dir.mkdir(parents=True, exist_ok=True)
            with (style_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (style_dir / "metrics.json").write_text(
                json.dumps({"summary": summarize(rows), "images": rows}, indent=2),
                encoding="utf-8",
            )
            contact_sheet(rows, style_dir / "contact_sheet.png", f"{style} {run_name}")
            summary["styles"][style] = summarize(rows)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[label] = summary
        print(run_dir / "summary.json")
    (output_root / f"{args.run_prefix}_all_strengths_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
