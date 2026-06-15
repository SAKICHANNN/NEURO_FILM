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
    parser.add_argument(
        "--mode",
        choices=("tone_locked", "wb_anchored", "legacy_rgb"),
        default="tone_locked",
        help=(
            "tone_locked preserves source RGB ratios for the tone pass; wb_anchored allows color residuals "
            "then anchors global R/G and B/G ratios; legacy_rgb applies the old RGB-delta response."
        ),
    )
    parser.add_argument("--strength", type=float, default=None, help="Legacy alias for --tone-strength in tone_locked mode.")
    parser.add_argument("--tone-strength", type=float, default=1.0)
    parser.add_argument(
        "--color-strength",
        type=float,
        default=0.0,
        help="Amount of residual Expert C color/WB response to add after the tone-locked pass.",
    )
    parser.add_argument(
        "--wb-anchor-strength",
        type=float,
        default=1.0,
        help="How strongly wb_anchored mode restores the source global R/G and B/G ratios.",
    )
    parser.add_argument(
        "--chroma-anchor-strength",
        type=float,
        default=0.0,
        help="How strongly the output chroma magnitude is limited toward the source chroma magnitude.",
    )
    parser.add_argument(
        "--chroma-headroom",
        type=float,
        default=0.10,
        help="Allowed chroma increase over source before chroma anchoring clamps the candidate.",
    )
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


def robust_channel_mean(rgb: np.ndarray) -> np.ndarray:
    lum = luma(rgb)
    mask = (lum > 0.05) & (lum < 0.95)
    if not np.any(mask):
        return rgb.reshape(-1, 3).mean(axis=0)
    return rgb[mask].reshape(-1, 3).mean(axis=0)


def interpolated_rgb_delta(raw_l: np.ndarray, curves: np.lib.npyio.NpzFile) -> np.ndarray:
    centers = curves["bin_centers"]
    rgb_delta = curves["rgb_delta_by_raw_luma"]
    delta_r = np.interp(raw_l, centers, rgb_delta[:, 0])
    delta_g = np.interp(raw_l, centers, rgb_delta[:, 1])
    delta_b = np.interp(raw_l, centers, rgb_delta[:, 2])
    return np.stack([delta_r, delta_g, delta_b], axis=2).astype(np.float32)


def apply_legacy_rgb_response(raw: np.ndarray, curves: np.lib.npyio.NpzFile, strength: float) -> np.ndarray:
    raw_l = luma(raw)
    delta = interpolated_rgb_delta(raw_l, curves)
    return np.clip(raw + delta * float(strength), 0.0, 1.0)


def apply_tone_locked_response(
    raw: np.ndarray,
    curves: np.lib.npyio.NpzFile,
    tone_strength: float,
    color_strength: float,
) -> np.ndarray:
    raw_l = luma(raw)
    centers = curves["bin_centers"]
    luma_delta = np.interp(raw_l, centers, curves["luma_delta_by_raw_luma"]).astype(np.float32)
    target_l = np.clip(raw_l + luma_delta * float(tone_strength), 0.0, 1.0)
    scale = target_l / np.maximum(raw_l, 1e-4)
    tone_locked = np.clip(raw * scale[..., None], 0.0, 1.0)

    if color_strength == 0:
        return tone_locked

    legacy_full = apply_legacy_rgb_response(raw, curves, tone_strength)
    residual = legacy_full - tone_locked
    return np.clip(tone_locked + residual * float(color_strength), 0.0, 1.0)


def anchor_white_balance(raw: np.ndarray, candidate: np.ndarray, anchor_strength: float) -> np.ndarray:
    raw_mean = robust_channel_mean(raw)
    candidate_mean = robust_channel_mean(candidate)
    raw_rg = raw_mean[0] / max(raw_mean[1], 1e-6)
    raw_bg = raw_mean[2] / max(raw_mean[1], 1e-6)
    candidate_rg = candidate_mean[0] / max(candidate_mean[1], 1e-6)
    candidate_bg = candidate_mean[2] / max(candidate_mean[1], 1e-6)

    strength = float(anchor_strength)
    gains = np.array(
        [
            (raw_rg / max(candidate_rg, 1e-6)) ** strength,
            1.0,
            (raw_bg / max(candidate_bg, 1e-6)) ** strength,
        ],
        dtype=np.float32,
    )
    adjusted = np.clip(candidate * gains[None, None, :], 0.0, 1.0)

    # Keep the already-computed tone response while removing mostly-global WB drift.
    candidate_l = luma(candidate)
    adjusted_l = luma(adjusted)
    relit = adjusted * (candidate_l / np.maximum(adjusted_l, 1e-4))[..., None]
    return np.clip(relit, 0.0, 1.0)


def anchor_chroma(raw: np.ndarray, candidate: np.ndarray, anchor_strength: float, headroom: float) -> np.ndarray:
    strength = float(anchor_strength)
    if strength <= 0:
        return candidate

    raw_l = luma(raw)
    candidate_l = luma(candidate)
    raw_vec = raw - raw_l[..., None]
    candidate_vec = candidate - candidate_l[..., None]
    raw_c = np.sqrt((raw_vec * raw_vec).sum(axis=2))
    candidate_c = np.sqrt((candidate_vec * candidate_vec).sum(axis=2))
    max_c = raw_c * (1.0 + float(headroom))
    target_scale = np.minimum(1.0, max_c / np.maximum(candidate_c, 1e-6))
    scale = 1.0 + (target_scale - 1.0) * strength
    guarded = candidate_l[..., None] + candidate_vec * scale[..., None]
    guarded = np.clip(guarded, 0.0, 1.0)

    # Re-apply candidate luminance after clipping/guarding to avoid accidental tone loss.
    guarded_l = luma(guarded)
    relit = guarded * (candidate_l / np.maximum(guarded_l, 1e-4))[..., None]
    return np.clip(relit, 0.0, 1.0)


def apply_wb_anchored_response(
    raw: np.ndarray,
    curves: np.lib.npyio.NpzFile,
    tone_strength: float,
    color_strength: float,
    wb_anchor_strength: float,
) -> np.ndarray:
    tone_locked = apply_tone_locked_response(raw, curves, tone_strength, 0.0)
    legacy_full = apply_legacy_rgb_response(raw, curves, tone_strength)
    candidate = np.clip(tone_locked + (legacy_full - tone_locked) * float(color_strength), 0.0, 1.0)
    return anchor_white_balance(raw, candidate, wb_anchor_strength)


def apply_response(raw: np.ndarray, curves: np.lib.npyio.NpzFile, args: argparse.Namespace) -> np.ndarray:
    tone_strength = args.strength if args.strength is not None else args.tone_strength
    if args.mode == "legacy_rgb":
        candidate = apply_legacy_rgb_response(raw, curves, tone_strength)
    elif args.mode == "wb_anchored":
        candidate = apply_wb_anchored_response(raw, curves, tone_strength, args.color_strength, args.wb_anchor_strength)
    else:
        candidate = apply_tone_locked_response(raw, curves, tone_strength, args.color_strength)
    candidate = anchor_chroma(raw, candidate, args.chroma_anchor_strength, args.chroma_headroom)
    if args.mode == "wb_anchored" and args.chroma_anchor_strength > 0:
        candidate = anchor_white_balance(raw, candidate, args.wb_anchor_strength)
    return candidate


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
        "baseline_red_green_ratio_delta": float(
            baseline[..., 0].mean() / (baseline[..., 1].mean() + 1e-6)
            - raw[..., 0].mean() / (raw[..., 1].mean() + 1e-6)
        ),
        "baseline_blue_green_ratio_delta": float(
            baseline[..., 2].mean() / (baseline[..., 1].mean() + 1e-6)
            - raw[..., 2].mean() / (raw[..., 1].mean() + 1e-6)
        ),
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
        "mode": args.mode,
        "strength": args.strength,
        "tone_strength": args.strength if args.strength is not None else args.tone_strength,
        "color_strength": args.color_strength,
        "wb_anchor_strength": args.wb_anchor_strength,
        "chroma_anchor_strength": args.chroma_anchor_strength,
        "chroma_headroom": args.chroma_headroom,
        "means": {
            "raw_target_luma_mae": mean("raw_target_luma_mae"),
            "baseline_target_luma_mae": mean("baseline_target_luma_mae"),
            "raw_target_rgb_mae": mean("raw_target_rgb_mae"),
            "baseline_target_rgb_mae": mean("baseline_target_rgb_mae"),
            "raw_target_chroma_mae": mean("raw_target_chroma_mae"),
            "baseline_target_chroma_mae": mean("baseline_target_chroma_mae"),
            "baseline_luma_delta_mean": mean("baseline_luma_delta_mean"),
            "baseline_chroma_delta_mean": mean("baseline_chroma_delta_mean"),
            "baseline_red_green_ratio_delta": mean("baseline_red_green_ratio_delta"),
            "baseline_blue_green_ratio_delta": mean("baseline_blue_green_ratio_delta"),
        },
        "note": (
            "Deterministic response baseline from compact stats. tone_locked preserves source RGB ratios "
            "for the tone pass; wb_anchored allows color residuals but restores global source WB ratios; "
            "chroma anchoring optionally limits saturation growth; legacy_rgb is the original failure baseline."
        ),
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
        baseline = apply_response(raw, curves, args)
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
