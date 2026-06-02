#!/usr/bin/env python3
"""Render several local-color-map strengths while reusing the safe-rich base."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_local_color_maps import (  # noqa: E402
    COLOR_STYLES,
    contact_sheet,
    load_image,
    save_rgb,
    source_paths,
    summarize,
)
from scripts.evaluate_render_safety import evaluate, load_rgb_u8, make_diff_map  # noqa: E402
from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.models.local_color_maps import apply_local_color_maps  # noqa: E402


DEFAULT_STRENGTHS = "s1p6:1.6,s2p2:2.2,s3p0:3.0,s4p0:4.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local bounded color maps at several strengths.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p50_gamutsafe" / "manifest.json",
    )
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "color_rendering_profiles.yaml")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "eval" / "color_engine_challenge")
    parser.add_argument("--prefix", default="local_maps_visual")
    parser.add_argument("--suffix", default="full")
    parser.add_argument("--styles", default=",".join(COLOR_STYLES))
    parser.add_argument("--strengths", default=DEFAULT_STRENGTHS)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_strengths(raw: str) -> list[tuple[str, float]]:
    pairs = []
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split(":", 1)
        pairs.append((label.strip(), float(value)))
    if not pairs:
        raise ValueError("At least one strength is required.")
    return pairs


def row_from_metrics(input_path: Path, after_path: Path, diff_path: Path, index: int, metrics: dict, map_metrics: dict) -> dict:
    return {
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


def main() -> int:
    args = parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    strengths = parse_strengths(args.strengths)
    paths = source_paths(args.source_manifest, args.limit)
    output_root = args.output_root.resolve()
    rows_by_run: dict[str, dict[str, list[dict]]] = {
        label: {style: [] for style in styles} for label, _ in strengths
    }
    summary_by_run = {
        label: {
            "engine": "local_bounded_maps",
            "source_manifest": str(args.source_manifest),
            "image_count": len(paths),
            "local_strength": strength,
            "styles": {},
        }
        for label, strength in strengths
    }

    for style in styles:
        profile = load_profile_values(args.profile_config, "safe-rich", style)
        guardrails = load_guardrail_config(args.guardrails, style)
        for index, input_path in enumerate(paths, start=1):
            image = load_image(input_path)
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
                guardrails=guardrails,
                dither=profile["dither"],
            )
            source_rgb = np.asarray(image, dtype=np.float32) / 255.0
            safe_rgb = np.asarray(safe_image, dtype=np.float32) / 255.0
            for label, strength in strengths:
                run_name = f"{args.prefix}_{label}_{args.suffix}" if args.suffix else f"{args.prefix}_{label}"
                style_dir = output_root / run_name / style
                local_rgb, map_metrics = apply_local_color_maps(
                    source_rgb,
                    safe_rgb,
                    style=style,
                    strength=strength,
                    output_margin=4,
                )
                after_path = style_dir / "after" / f"{index:02d}_{style}_local_maps_{label}.png"
                diff_path = style_dir / "diff_maps" / f"{index:02d}_{style}_diff.png"
                save_rgb(local_rgb, after_path)
                metrics = evaluate(input_path, after_path)
                make_diff_map(load_rgb_u8(input_path), load_rgb_u8(after_path), diff_path)
                row = row_from_metrics(input_path, after_path, diff_path, index, metrics, map_metrics)
                rows_by_run[label][style].append(row)
                print(f"{label} {style} {index:02d} Lssim={row['l_ssim']:.5f} chroma={row['after_chroma_mean']:.2f}")

    for label, _strength in strengths:
        run_name = f"{args.prefix}_{label}_{args.suffix}" if args.suffix else f"{args.prefix}_{label}"
        run_dir = output_root / run_name
        for style in styles:
            rows = rows_by_run[label][style]
            style_dir = run_dir / style
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
            summary_by_run[label]["styles"][style] = summarize(rows)
        (run_dir / "summary.json").write_text(json.dumps(summary_by_run[label], indent=2), encoding="utf-8")
        print(run_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
