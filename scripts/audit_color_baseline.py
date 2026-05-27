#!/usr/bin/env python3
"""Audit the current deterministic color baseline on configured eval sources."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_render_safety import evaluate, load_rgb_u8, make_diff_map  # noqa: E402
from scripts.pipeline_color_baseline import load_image, style_transfer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline renders and safety metrics over eval images.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "eval" / "baseline_current")
    parser.add_argument("--styles", default="all", help="Comma-separated styles or all.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--luma-strength", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=0.012)
    parser.add_argument("--gamut-safe", action="store_true", help="Audit the existing optional gamut-safe flag.")
    parser.add_argument("--gamut-mode", choices=("off", "source", "chroma"), default=None)
    parser.add_argument("--tone-rolloff", type=float, default=0.0)
    parser.add_argument("--output-margin", type=int, default=0)
    parser.add_argument("--use-guardrails", action="store_true")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--neutral-protect", type=float, default=None)
    parser.add_argument("--skin-protect", type=float, default=None)
    parser.add_argument("--max-chroma-gain", type=float, default=None)
    parser.add_argument("--max-chroma-boost", type=float, default=None)
    parser.add_argument("--max-chroma-absolute", type=float, default=None)
    parser.add_argument("--dither", type=float, default=None)
    return parser.parse_args()


def slug_float(value: float) -> str:
    return f"{value:.3g}".replace(".", "p")


def load_source_rows(path: Path, limit: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        rows = rows[:limit]
    result = []
    for index, row in enumerate(rows, start=1):
        before = Path(row.get("before") or row.get("input") or "")
        if before.exists():
            result.append({"id": f"{index:02d}", "before": before, "source_row": row})
    return result


def parse_styles(raw: str, available: list[str]) -> list[str]:
    if raw == "all":
        return available
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [style for style in requested if style not in available]
    if unknown:
        raise ValueError(f"Unknown styles: {unknown}")
    return requested


def make_style_contact_sheet(rows: list[dict], output_path: Path, title: str) -> None:
    font = ImageFont.load_default()
    tiles = []
    for row in rows:
        before = Image.open(row["before"]).convert("RGB")
        after = Image.open(row["after"]).convert("RGB")
        diff = Image.open(row["diff_map"]).convert("RGB")
        for image in (before, after, diff):
            image.thumbnail((180, 150), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (540, 184), "white")
        draw = ImageDraw.Draw(tile)
        label = f"{row['id']} clip={row['new_clipped_pixel_count']} Lssim={row['l_ssim']:.5f}"
        draw.text((8, 8), label, fill="black", font=font)
        for column, image in enumerate((before, after, diff)):
            tile.paste(image, (column * 180 + (180 - image.width) // 2, 34))
        tiles.append(tile)

    cols = 1
    title_h = 34
    sheet = Image.new("RGB", (540 * cols, len(tiles) * 184 + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill="black", font=font)
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, title_h + index * 184))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")


def summarize_style(rows: list[dict]) -> dict:
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


def main() -> int:
    args = parse_args()
    stats_doc = json.loads(args.stats.read_text(encoding="utf-8"))
    styles = parse_styles(args.styles, sorted(stats_doc["styles"]))
    sources = load_source_rows(args.source_manifest, args.limit)
    if not sources:
        raise ValueError(f"No source images found from {args.source_manifest}")

    run_id = f"s{slug_float(args.strength)}_l{slug_float(args.luma_strength)}_g{slug_float(args.grain)}"
    if args.gamut_safe:
        run_id += "_gamutsafe"
    if args.output_margin > 0:
        run_id += f"_m{args.output_margin}"
    if args.use_guardrails:
        run_id += "_guards"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_manifest": str(args.source_manifest),
        "style_count": len(styles),
        "image_count": len(sources),
        "strength": args.strength,
        "luma_strength": args.luma_strength,
        "grain": args.grain,
        "gamut_safe": args.gamut_safe,
        "gamut_mode": args.gamut_mode or ("source" if args.gamut_safe else "off"),
        "tone_rolloff": args.tone_rolloff,
        "output_margin": args.output_margin,
        "guardrails": args.use_guardrails,
        "styles": {},
    }

    for style in styles:
        guardrail_config = (
            json.loads(args.guardrails.read_text(encoding="utf-8")).get("defaults", {}) if args.use_guardrails else {}
        )
        if args.use_guardrails:
            guardrail_doc = json.loads(args.guardrails.read_text(encoding="utf-8"))
            guardrail_config = dict(guardrail_doc.get("defaults", {}))
            guardrail_config.update(guardrail_doc.get("styles", {}).get(style, {}))
        style_dir = output_dir / style
        after_dir = style_dir / "after"
        diff_dir = style_dir / "diff_maps"
        after_dir.mkdir(parents=True, exist_ok=True)
        diff_dir.mkdir(parents=True, exist_ok=True)
        metric_rows: list[dict] = []
        manifest_rows: list[dict] = []

        for source in sources:
            before_path = source["before"]
            before_image = load_image(before_path)
            after = style_transfer(
                before_image,
                stats_doc["styles"][style],
                style,
                strength=args.strength,
                luma_strength=args.luma_strength,
                grain=args.grain,
                seed=42 + int(source["id"]),
                gamut_safe=args.gamut_safe,
                gamut_mode=args.gamut_mode,
                tone_rolloff=args.tone_rolloff,
                output_margin=args.output_margin,
                guardrails=guardrail_config,
                neutral_protect=args.neutral_protect,
                skin_protect=args.skin_protect,
                max_chroma_gain=args.max_chroma_gain,
                max_chroma_boost=args.max_chroma_boost,
                max_chroma_absolute=args.max_chroma_absolute,
                dither=args.dither,
            )
            after_path = after_dir / f"{source['id']}_{style}_{run_id}.png"
            diff_path = diff_dir / f"{source['id']}_{style}_{run_id}_diff.png"
            after.save(after_path, "PNG")
            metrics = evaluate(before_path, after_path)
            make_diff_map(load_rgb_u8(before_path), load_rgb_u8(after_path), diff_path)
            row = {
                "id": source["id"],
                "before": str(before_path),
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
            }
            metric_rows.append(row)
            manifest_rows.append({**row, "style": style, "run_id": run_id})
            print(f"{style} {source['id']} Lssim={row['l_ssim']:.5f} new_clip={row['new_clipped_pixel_count']}")

        fields = list(manifest_rows[0])
        with (style_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest_rows)
        (style_dir / "metrics.json").write_text(
            json.dumps({"summary": summarize_style(metric_rows), "images": metric_rows}, indent=2),
            encoding="utf-8",
        )
        make_style_contact_sheet(metric_rows, style_dir / "contact_sheet.png", f"{style} current baseline {run_id}")
        summary["styles"][style] = summarize_style(metric_rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
