#!/usr/bin/env python3
"""Compare mined real-photo halo statistics with current halation presets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mine_halation_patches import patch_metrics, save_rgb  # noqa: E402
from src.filmfx import build_physical_halation_layer, composite_layers, get_halation_preset, list_halation_presets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-photo halation alignment.")
    parser.add_argument("--metrics", type=Path, default=Path("outputs/eval/halation_real_photo_v1/metrics/real_patch_metrics.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eval/halation_real_photo_v1"))
    parser.add_argument("--presets", nargs="*", default=["vision3_clean", "vision3_push", "cinestill_balanced", "cinestill_strong", "cinestill_amber", "classic_soft"])
    parser.add_argument("--max-real-patches", type=int, default=12)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def layer_on_black(layer) -> np.ndarray:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(layer.rgb * alpha, 0.0, 1.0)


def layer_on_white(layer) -> np.ndarray:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(1.0 - alpha + layer.rgb * alpha, 0.0, 1.0)


def synthetic_source(metric: dict, size: int = 192) -> tuple[np.ndarray, tuple[float, float], float]:
    background = float(np.clip(metric.get("background_luma", 0.08), 0.0, 0.55))
    source_peak = float(np.clip(metric.get("source_luma_peak", 0.96), background + 0.20, 1.0))
    source_radius = float(np.clip(metric.get("source_radius_px", 4.0), 2.0, 18.0))
    cx = cy = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy)
    source = np.exp(-0.5 * (dist / max(source_radius, 1.0)) ** 2)
    rgb = np.full((size, size, 3), background, dtype=np.float32)
    rgb += (source_peak - background) * source[..., None]
    return np.clip(rgb, 0.0, 1.0), (cx, cy), source_radius


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def summarize_metrics(metrics: list[dict]) -> dict[str, float]:
    keys = [
        "visible_radius_px",
        "radius_normalized",
        "red_green_ratio_outer",
        "blue_leakage",
        "orange_core_score",
        "background_luma",
        "halo_confidence",
    ]
    return {f"median_{key}": median([float(row.get(key, 0.0)) for row in metrics]) for key in keys}


def distribution_distance(real: list[dict], simulated: list[dict], key: str) -> float:
    real_values = np.asarray([float(row.get(key, 0.0)) for row in real], dtype=np.float32)
    sim_values = np.asarray([float(row.get(key, 0.0)) for row in simulated], dtype=np.float32)
    if real_values.size == 0 or sim_values.size == 0:
        return 0.0
    return float(abs(np.median(real_values) - np.median(sim_values)) / max(np.std(real_values) + 1e-4, 0.05))


def make_alignment_sheet(rows: list[dict], output: Path) -> None:
    font = ImageFont.load_default()
    tile_w = 170
    tile_h = 170
    label_h = 50
    columns = ["real", "synthetic", "layer black", "layer white", "combined"]
    rendered_rows: list[Image.Image] = []
    for row in rows:
        tile = Image.new("RGB", (tile_w * len(columns), tile_h + label_h), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((6, 6), f"{row['patch_id']} | {row['preset']}", fill="black", font=font)
        images = [row["real"], row["synthetic"], row["black"], row["white"], row["combined"]]
        for idx, image_array in enumerate(images):
            img = Image.fromarray(np.rint(np.clip(image_array, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
            img.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            tile.paste(img, (idx * tile_w + (tile_w - img.width) // 2, label_h))
            draw.text((idx * tile_w + 5, 25), columns[idx], fill="black", font=font)
        rendered_rows.append(tile)
    if not rendered_rows:
        return
    sheet = Image.new("RGB", (rendered_rows[0].width, len(rendered_rows) * rendered_rows[0].height + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), "real-photo vs current simulator alignment", fill="black", font=font)
    for index, tile in enumerate(rendered_rows):
        sheet.paste(tile, (0, 30 + index * tile.height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    metrics_dir = output_root / "metrics"
    patch_dir = output_root / "patches" / "simulated"
    contact_dir = output_root / "contact_sheets"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    real_metrics: list[dict] = json.loads(args.metrics.read_text(encoding="utf-8"))
    real_metrics = real_metrics[: args.max_real_patches]
    simulated: list[dict] = []
    sheet_rows: list[dict] = []
    for real in real_metrics:
        real_patch = load_rgb(Path(real["patch_path"]))
        synthetic, center, source_radius = synthetic_source(real, size=real_patch.shape[0])
        for preset_id in args.presets:
            controls = get_halation_preset(preset_id).controls
            layer = build_physical_halation_layer(synthetic, controls)
            combined = composite_layers(synthetic, [layer], output_margin=0)
            sim_metric = patch_metrics(combined, center, source_radius)
            sim_metric.update(
                {
                    "real_patch_id": real["patch_id"],
                    "preset": preset_id,
                    "controls_evidence_level": "uncalibrated_heuristic",
                }
            )
            simulated.append(sim_metric)
            if len(sheet_rows) < 18:
                base = f"{real['patch_id']}_{preset_id}"
                save_rgb(synthetic, patch_dir / f"{base}_synthetic.png")
                save_rgb(layer_on_black(layer), patch_dir / f"{base}_layer_black.png")
                save_rgb(layer_on_white(layer), patch_dir / f"{base}_layer_white.png")
                save_rgb(combined, patch_dir / f"{base}_combined.png")
                sheet_rows.append(
                    {
                        "patch_id": real["patch_id"],
                        "preset": preset_id,
                        "real": real_patch,
                        "synthetic": synthetic,
                        "black": layer_on_black(layer),
                        "white": layer_on_white(layer),
                        "combined": combined,
                    }
                )
    simulated_path = metrics_dir / "simulated_patch_metrics.json"
    simulated_path.write_text(json.dumps(simulated, indent=2), encoding="utf-8")
    by_preset: dict[str, list[dict]] = {}
    for row in simulated:
        by_preset.setdefault(row["preset"], []).append(row)
    preset_summary = {}
    for preset_id, rows in by_preset.items():
        preset_summary[preset_id] = {
            **summarize_metrics(rows),
            "distribution_distance_radius": distribution_distance(real_metrics, rows, "visible_radius_px"),
            "distribution_distance_hue": distribution_distance(real_metrics, rows, "red_green_ratio_outer"),
            "blue_leakage_delta": median([float(row["blue_leakage"]) for row in rows])
            - median([float(row["blue_leakage"]) for row in real_metrics]),
            "patch_count": len(rows),
        }
    suggestions = []
    real_radius = median([float(row["visible_radius_px"]) for row in real_metrics])
    real_hue = median([float(row["red_green_ratio_outer"]) for row in real_metrics])
    for preset_id, summary in preset_summary.items():
        sim_radius = summary["median_visible_radius_px"]
        sim_hue = summary["median_red_green_ratio_outer"]
        if sim_radius < real_radius * 0.70:
            suggestions.append({"preset": preset_id, "suggestion": "simulated visible radius is low; test higher diffusion before changing amount"})
        if sim_hue < real_hue * 0.85:
            suggestions.append({"preset": preset_id, "suggestion": "outer red/green ratio is low; test warmer color_response or warm_core"})
        if summary["blue_leakage_delta"] > 0.12:
            suggestions.append({"preset": preset_id, "suggestion": "blue leakage high; preserve current red/orange or density-family constraints"})
    alignment = {
        "evidence_level": "measured display-level unpaired statistics",
        "real_patch_count": len(real_metrics),
        "simulated_patch_count": len(simulated),
        "real_summary": summarize_metrics(real_metrics),
        "preset_summary": preset_summary,
        "limitations": [
            "Web JPEGs are not scene-linear scans and cannot provide physical stock constants.",
            "Commons stock labels are treated as user-tagged source claims unless stated on the file page.",
            "The simulator comparison uses synthetic sources matched to real patch display statistics, not paired originals.",
        ],
    }
    (metrics_dir / "alignment_summary.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    (metrics_dir / "parameter_suggestions.json").write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
    make_alignment_sheet(sheet_rows, contact_dir / "alignment_contact_sheet.png")
    report = [
        "# Halation Real-Photo Validation V1",
        "",
        "Evidence level: measured display-level, unpaired statistics.",
        "",
        f"- Real patches analyzed: {len(real_metrics)}",
        f"- Simulated preset patches: {len(simulated)}",
        f"- Contact sheet: `{(contact_dir / 'alignment_contact_sheet.png').as_posix()}`",
        f"- Real patch sheet: `{(contact_dir / 'real_patches_contact_sheet.png').as_posix()}`",
        "",
        "## Real Summary",
        "",
        "```json",
        json.dumps(alignment["real_summary"], indent=2),
        "```",
        "",
        "## Preset Summary",
        "",
        "```json",
        json.dumps(preset_summary, indent=2),
        "```",
        "",
        "## Parameter Suggestions",
        "",
        "```json",
        json.dumps(suggestions, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- This is not true film-physics calibration.",
        "- No default preset was changed by this script.",
        "- Real-photo contact sheets remain under ignored `outputs/`.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(metrics_dir / "alignment_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
