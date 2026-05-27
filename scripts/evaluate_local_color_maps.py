#!/usr/bin/env python3
"""Evaluate local bounded color maps against the safe-rich baseline."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_render_safety import evaluate, load_rgb_u8, make_diff_map  # noqa: E402
from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.models.local_color_maps import apply_local_color_maps  # noqa: E402


COLOR_STYLES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local bounded color maps.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "color_rendering_profiles.yaml")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eval" / "color_engine_challenge" / "local_maps")
    parser.add_argument("--styles", default=",".join(COLOR_STYLES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-strength", type=float, default=1.0)
    return parser.parse_args()


def source_paths(path: Path, limit: int) -> list[Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    return [Path(row.get("before") or row.get("input")) for row in rows if Path(row.get("before") or row.get("input")).exists()]


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


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
    }


def contact_sheet(rows: list[dict], path: Path, title: str) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for row in rows:
        before = Image.open(row["before"]).convert("RGB")
        after = Image.open(row["after"]).convert("RGB")
        for image in (before, after):
            image.thumbnail((220, 170), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (440, 204), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"{row['id']} L={row['l_ssim']:.5f} C={row['after_chroma_mean']:.2f}", fill="black", font=font)
        tile.paste(before, ((220 - before.width) // 2, 34))
        tile.paste(after, (220 + (220 - after.width) // 2, 34))
        thumbs.append(tile)
    sheet = Image.new("RGB", (440, len(thumbs) * 204 + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black", font=font)
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, (0, 30 + index * 204))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")


def main() -> int:
    args = parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    paths = source_paths(args.source_manifest, args.limit)
    output_dir = args.output_dir.resolve()
    summary = {
        "engine": "local_bounded_maps",
        "source_manifest": str(args.source_manifest),
        "image_count": len(paths),
        "local_strength": args.local_strength,
        "styles": {},
    }
    for style in styles:
        rows: list[dict] = []
        style_dir = output_dir / style
        for index, input_path in enumerate(paths, start=1):
            image = load_image(input_path)
            profile = load_profile_values(args.profile_config, "safe-rich", style)
            safe_image = style_transfer(
                image,
                stats["styles"][style],
                style,
                strength=profile["strength"],
                luma_strength=profile["luma_strength"],
                grain=profile["grain"],
                seed=7 + index,
                gamut_safe=profile["gamut_safe"],
                gamut_mode=profile["gamut_mode"],
                tone_rolloff=profile["tone_rolloff"],
                shadow_floor_l=profile["shadow_floor_l"],
                highlight_ceiling_l=profile["highlight_ceiling_l"],
                preserve_luma_detail_strength=profile["preserve_luma_detail"],
                chroma_curve_strength=profile["chroma_curve_strength"],
                output_margin=profile["output_margin"],
                guardrails=load_guardrail_config(args.guardrails, style),
                dither=profile["dither"],
            )
            source_rgb = np.asarray(image, dtype=np.float32) / 255.0
            safe_rgb = np.asarray(safe_image, dtype=np.float32) / 255.0
            local_rgb, map_metrics = apply_local_color_maps(
                source_rgb,
                safe_rgb,
                style=style,
                strength=args.local_strength,
                output_margin=4,
            )
            after_path = style_dir / "after" / f"{index:02d}_{style}_local_maps.png"
            diff_path = style_dir / "diff_maps" / f"{index:02d}_{style}_diff.png"
            save_rgb(local_rgb, after_path)
            metrics = evaluate(input_path, after_path)
            make_diff_map(load_rgb_u8(input_path), load_rgb_u8(after_path), diff_path)
            row = {
                "id": f"{index:02d}",
                "before": str(input_path),
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
                **map_metrics,
            }
            rows.append(row)
            print(f"{style} {index:02d} Lssim={row['l_ssim']:.5f} chroma={row['after_chroma_mean']:.2f}")
        style_dir.mkdir(parents=True, exist_ok=True)
        with (style_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (style_dir / "metrics.json").write_text(json.dumps({"summary": summarize(rows), "images": rows}, indent=2), encoding="utf-8")
        contact_sheet(rows, style_dir / "contact_sheet.png", f"{style} local bounded maps")
        summary["styles"][style] = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
