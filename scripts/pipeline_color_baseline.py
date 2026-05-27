#!/usr/bin/env python3
"""Content-safe deterministic film color baseline.

This pipeline never uses a generative model. It transfers robust Lab color
statistics from the film-domain dataset and optionally adds light grain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab


ROOT = Path(__file__).resolve().parents[1]
B_AND_W_STYLES = {"hp5", "tri_x_400"}
DEFAULT_GUARDRAILS = ROOT / "configs" / "color_guardrails.json"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply deterministic film color transfer.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--style", default="portra_800")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--strengths", default=None, help="Comma-separated grid, e.g. 0.35,0.55,0.75")
    parser.add_argument("--luma-strength", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=0.012)
    parser.add_argument("--gamut-safe", action="store_true", help="Compress Lab transform into displayable sRGB gamut.")
    parser.add_argument(
        "--gamut-mode",
        choices=("off", "source", "chroma"),
        default=None,
        help="Gamut compression mode. --gamut-safe defaults to source.",
    )
    parser.add_argument("--tone-rolloff", type=float, default=0.0, help="Monotonic Lab L roll-off strength in [0, 1].")
    parser.add_argument("--shadow-floor-l", type=float, default=1.0)
    parser.add_argument("--highlight-ceiling-l", type=float, default=99.0)
    parser.add_argument("--preserve-luma-detail", type=float, default=0.0)
    parser.add_argument("--chroma-curve-strength", type=float, default=0.0)
    parser.add_argument("--output-margin", type=int, default=0, help="Reserve 8-bit output headroom, e.g. 4 -> [4, 251].")
    parser.add_argument("--format", choices=("auto", "png", "jpeg"), default="auto")
    parser.add_argument("--fail-on-clip", action="store_true", help="Exit nonzero if the saved output has new hard clipping.")
    parser.add_argument("--use-guardrails", action="store_true", help="Apply neutral/skin/chroma guardrails.")
    parser.add_argument("--guardrails", type=Path, default=DEFAULT_GUARDRAILS)
    parser.add_argument("--neutral-protect", type=float, default=None)
    parser.add_argument("--skin-protect", type=float, default=None)
    parser.add_argument("--max-chroma-gain", type=float, default=None)
    parser.add_argument("--max-chroma-boost", type=float, default=None)
    parser.add_argument("--max-chroma-absolute", type=float, default=None)
    parser.add_argument("--dither", type=float, default=None, help="8-bit quantization dither strength in LSBs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "color_baseline")
    parser.add_argument("--contact-sheet", action="store_true")
    return parser.parse_args()


def slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def lab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.float32)
    y = (lab[..., 0] + 16.0) / 116.0
    x = y + lab[..., 1] / 500.0
    z = y - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def inv_f(value: np.ndarray) -> np.ndarray:
        return np.where(value > delta, value**3, 3.0 * delta**2 * (value - 4.0 / 29.0))

    xyz = np.stack([0.95047 * inv_f(x), inv_f(y), 1.08883 * inv_f(z)], axis=-1)
    matrix = np.asarray(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    return xyz @ matrix.T


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    return np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(np.maximum(linear_rgb, 0.0), 1.0 / 2.4) - 0.055,
    )


def in_srgb_gamut(lab: np.ndarray) -> np.ndarray:
    linear = lab_to_linear_srgb(lab)
    return np.all((linear >= 0.0) & (linear <= 1.0), axis=-1)


def compress_to_srgb_gamut(source_lab: np.ndarray, target_lab: np.ndarray, iterations: int = 14) -> np.ndarray:
    """Binary-search each pixel along source->target until it stays in sRGB gamut."""
    low = np.zeros(source_lab.shape[:2] + (1,), dtype=np.float32)
    high = np.ones_like(low)
    delta = target_lab - source_lab
    for _ in range(iterations):
        mid = (low + high) * 0.5
        candidate = source_lab + delta * mid
        valid = in_srgb_gamut(candidate)[..., None]
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    return source_lab + delta * low


def compress_chroma_to_srgb_gamut(target_lab: np.ndarray, iterations: int = 14) -> np.ndarray:
    """Reduce Lab chroma at fixed L and hue until every pixel fits sRGB."""
    low = np.zeros(target_lab.shape[:2] + (1,), dtype=np.float32)
    high = np.ones_like(low)
    neutral = target_lab.copy()
    neutral[..., 1:] = 0.0
    delta = target_lab - neutral
    for _ in range(iterations):
        mid = (low + high) * 0.5
        candidate = neutral + delta * mid
        valid = in_srgb_gamut(candidate)[..., None]
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    return neutral + delta * low


def apply_tone_rolloff(lab: np.ndarray, strength: float, shadow_floor_l: float, highlight_ceiling_l: float) -> np.ndarray:
    if strength <= 0:
        return lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = lab.copy()
    luminance = np.clip(out[..., 0] / 100.0, 0.0, 1.0)
    smooth = luminance * luminance * (3.0 - 2.0 * luminance)
    low = np.clip(shadow_floor_l / 100.0, 0.0, 0.25)
    high = np.clip(highlight_ceiling_l / 100.0, 0.75, 1.0)
    rolled = low + smooth * (high - low)
    out[..., 0] = 100.0 * ((1.0 - strength) * luminance + strength * rolled)
    return out


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def skin_like_mask(lab: np.ndarray) -> np.ndarray:
    return (
        (lab[..., 0] > 20.0)
        & (lab[..., 0] < 92.0)
        & (lab[..., 1] > 4.0)
        & (lab[..., 1] < 28.0)
        & (lab[..., 2] > 4.0)
        & (lab[..., 2] < 46.0)
    )


def apply_color_guardrails(
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    neutral_protect: float,
    skin_protect: float,
    max_chroma_gain: float | None,
    max_chroma_boost: float | None,
    max_chroma_absolute: float | None,
) -> np.ndarray:
    out = target_lab.copy()
    source_ab = source_lab[..., 1:3]
    out_ab = out[..., 1:3]
    source_chroma = np.linalg.norm(source_ab, axis=2, keepdims=True)
    out_chroma = np.linalg.norm(out_ab, axis=2, keepdims=True)

    if neutral_protect > 0:
        neutral_weight = 1.0 - smoothstep(4.0, 14.0, source_chroma)
        blend = np.clip(neutral_weight * neutral_protect, 0.0, 1.0)
        out_ab = out_ab * (1.0 - blend) + source_ab * blend

    if skin_protect > 0:
        skin_weight = skin_like_mask(source_lab)[..., None].astype(np.float32) * np.clip(skin_protect, 0.0, 1.0)
        out_ab = out_ab * (1.0 - skin_weight) + source_ab * skin_weight

    if max_chroma_gain is not None or max_chroma_boost is not None or max_chroma_absolute is not None:
        gain = float(max_chroma_gain if max_chroma_gain is not None else 999.0)
        boost = float(max_chroma_boost if max_chroma_boost is not None else 999.0)
        cap = np.maximum(source_chroma * gain, source_chroma + boost)
        if max_chroma_absolute is not None:
            cap = np.minimum(cap, float(max_chroma_absolute))
        out_chroma = np.maximum(np.linalg.norm(out_ab, axis=2, keepdims=True), 1e-6)
        scale = np.minimum(1.0, cap / out_chroma)
        # Ease into the cap over the upper chroma range to avoid an obvious hard shelf.
        knee = smoothstep(0.82, 1.0, out_chroma / np.maximum(cap, 1e-6))
        out_ab = out_ab * ((1.0 - knee) + knee * scale)

    out[..., 1:3] = out_ab
    return out


def apply_chroma_curve(source_lab: np.ndarray, target_lab: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return target_lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = target_lab.copy()
    source_chroma = np.linalg.norm(source_lab[..., 1:3], axis=2, keepdims=True)
    saturated_weight = smoothstep(24.0, 60.0, source_chroma)
    scale = 1.0 - strength * saturated_weight
    out[..., 1:3] = source_lab[..., 1:3] + (out[..., 1:3] - source_lab[..., 1:3]) * scale
    return out


def preserve_luma_detail(source_lab: np.ndarray, target_lab: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return target_lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = target_lab.copy()
    source_l = source_lab[..., 0]
    target_l = target_lab[..., 0]
    source_detail = source_l - gaussian_filter(source_l, sigma=1.1)
    target_base = gaussian_filter(target_l, sigma=1.1)
    target_detail = target_l - target_base
    out[..., 0] = np.clip(target_base + target_detail * (1.0 - strength) + source_detail * strength, 0.0, 100.0)
    return out


def lab_to_rgb_no_clip(lab: np.ndarray) -> np.ndarray:
    linear = lab_to_linear_srgb(lab)
    # Gamut-safe callers should already be inside bounds; this guards tiny float error only.
    return np.clip(linear_to_srgb(np.clip(linear, 0.0, 1.0)), 0.0, 1.0)


def apply_output_margin(rgb: np.ndarray, output_margin: int) -> np.ndarray:
    if output_margin <= 0:
        return rgb
    margin = max(0, min(32, int(output_margin)))
    low = margin / 255.0
    high = 1.0 - low
    return np.clip(rgb, low, high)


def load_guardrail_config(path: Path, style: str) -> dict:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    config = dict(doc.get("defaults", {}))
    config.update(doc.get("styles", {}).get(style, {}))
    return config


def resolve_guardrail_value(config: dict, cli_value: float | None, key: str) -> float | None:
    return cli_value if cli_value is not None else config.get(key)


def style_transfer(
    image: Image.Image,
    stats: dict,
    style: str,
    strength: float,
    luma_strength: float,
    grain: float,
    seed: int,
    gamut_safe: bool,
    gamut_mode: str | None = None,
    tone_rolloff: float = 0.0,
    shadow_floor_l: float = 1.0,
    highlight_ceiling_l: float = 99.0,
    preserve_luma_detail_strength: float = 0.0,
    chroma_curve_strength: float = 0.0,
    output_margin: int = 0,
    guardrails: dict | None = None,
    neutral_protect: float | None = None,
    skin_protect: float | None = None,
    max_chroma_gain: float | None = None,
    max_chroma_boost: float | None = None,
    max_chroma_absolute: float | None = None,
    dither: float | None = None,
) -> Image.Image:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb)
    src_mean = lab.reshape(-1, 3).mean(axis=0)
    src_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 1e-3)
    dst_mean = np.asarray(stats["mean"], dtype=np.float32)
    dst_std = np.asarray(stats["std"], dtype=np.float32)

    transferred = (lab - src_mean) / src_std * dst_std + dst_mean
    out = lab.copy()
    out[..., 0] = lab[..., 0] + luma_strength * strength * (transferred[..., 0] - lab[..., 0])
    out[..., 1:] = lab[..., 1:] + strength * (transferred[..., 1:] - lab[..., 1:])

    if style in B_AND_W_STYLES:
        gray_strength = min(1.0, strength * 1.35)
        out[..., 1:] *= 1.0 - gray_strength
        contrast = 1.0 + 0.30 * strength
        out[..., 0] = np.clip((out[..., 0] - 50.0) * contrast + 50.0, 0.0, 100.0)

    out = apply_chroma_curve(lab, out, chroma_curve_strength)
    out = preserve_luma_detail(lab, out, preserve_luma_detail_strength)
    guardrails = guardrails or {}
    neutral_protect = resolve_guardrail_value(guardrails, neutral_protect, "neutral_protect") or 0.0
    skin_protect = resolve_guardrail_value(guardrails, skin_protect, "skin_protect") or 0.0
    max_chroma_gain = resolve_guardrail_value(guardrails, max_chroma_gain, "max_chroma_gain")
    max_chroma_boost = resolve_guardrail_value(guardrails, max_chroma_boost, "max_chroma_boost")
    max_chroma_absolute = resolve_guardrail_value(guardrails, max_chroma_absolute, "max_chroma_absolute")
    out = apply_color_guardrails(
        lab,
        out,
        neutral_protect=float(neutral_protect),
        skin_protect=float(skin_protect),
        max_chroma_gain=max_chroma_gain,
        max_chroma_boost=max_chroma_boost,
        max_chroma_absolute=max_chroma_absolute,
    )
    out = apply_tone_rolloff(out, tone_rolloff, shadow_floor_l, highlight_ceiling_l)

    resolved_gamut_mode = gamut_mode or ("source" if gamut_safe else "off")
    if resolved_gamut_mode == "source":
        out = compress_to_srgb_gamut(lab, out)
        result = lab_to_rgb_no_clip(out)
    elif resolved_gamut_mode == "chroma":
        out = compress_chroma_to_srgb_gamut(out)
        result = lab_to_rgb_no_clip(out)
    else:
        out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
        out[..., 1:] = np.clip(out[..., 1:], -128.0, 127.0)
        result = lab2rgb(out)

    if grain > 0:
        rng = np.random.default_rng(seed)
        luminance = result.mean(axis=2, keepdims=True)
        noise_scale = grain * (0.55 + 0.9 * (1.0 - luminance))
        noise = rng.normal(0.0, noise_scale, size=result.shape).astype(np.float32)
        result = np.clip(result + noise, 0.0, 1.0)

    dither = resolve_guardrail_value(guardrails, dither, "dither") or 0.0
    if dither > 0:
        rng = np.random.default_rng(seed + 1009)
        result = np.clip(result + rng.uniform(-0.5, 0.5, size=result.shape).astype(np.float32) * (float(dither) / 255.0), 0.0, 1.0)

    result = apply_output_margin(result, output_margin)
    return Image.fromarray(np.rint(np.clip(result * 255.0, 0, 255)).astype(np.uint8), mode="RGB")


def infer_save_format(path: Path, requested: str) -> str:
    if requested == "png":
        return "PNG"
    if requested == "jpeg":
        return "JPEG"
    if path.suffix.lower() == ".png":
        return "PNG"
    return "JPEG"


def save_output(image: Image.Image, path: Path, requested_format: str) -> None:
    image_format = infer_save_format(path, requested_format)
    if image_format == "PNG":
        image.save(path, "PNG")
    else:
        image.save(path, "JPEG", quality=95)


def saved_output_has_new_clip(before: Image.Image, output_path: Path, output_margin: int) -> tuple[bool, dict]:
    before_arr = np.asarray(before.convert("RGB"), dtype=np.uint8)
    after_arr = np.asarray(load_image(output_path), dtype=np.uint8)
    before_clip = np.any((before_arr <= 0) | (before_arr >= 255), axis=2)
    after_clip = np.any((after_arr <= 0) | (after_arr >= 255), axis=2)
    new_clip = after_clip & ~before_clip
    margin = max(0, int(output_margin))
    within_margin = True
    if margin > 0:
        within_margin = bool(after_arr.min() >= margin and after_arr.max() <= 255 - margin)
    stats = {
        "after_min": int(after_arr.min()),
        "after_max": int(after_arr.max()),
        "new_clipped_pixel_count": int(new_clip.sum()),
        "within_output_margin": within_margin,
    }
    return bool(new_clip.any() or not within_margin), stats


def write_contact_sheet(rows: list[dict], output_dir: Path, label: str) -> Path:
    font = ImageFont.load_default()
    thumbs = []
    for row in rows:
        image = Image.open(row["output"]).convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (256, 286), "white")
        canvas.paste(image, ((256 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 264), row["label"], fill="black", font=font)
        thumbs.append(canvas)

    cols = min(4, len(thumbs))
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows_count * 286 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), label, fill="black", font=font)
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 256, 28 + (idx // cols) * 286))
    path = output_dir / "contact_sheets" / f"{label}_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return path


def main() -> int:
    args = parse_args()
    stats_doc = json.loads(args.stats.read_text(encoding="utf-8"))
    styles = sorted(stats_doc["styles"]) if args.all else [args.style]
    strengths = parse_float_list(args.strengths) if args.strengths else [args.strength]
    image = load_image(args.input)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for style in styles:
        if style not in stats_doc["styles"]:
            raise ValueError(f"Style not found in stats: {style}")
        guardrail_config = load_guardrail_config(args.guardrails, style) if args.use_guardrails else {}
        style_rows = []
        for strength in strengths:
            out = style_transfer(
                image,
                stats_doc["styles"][style],
                style,
                strength=strength,
                luma_strength=args.luma_strength,
                grain=args.grain,
                seed=args.seed,
                gamut_safe=args.gamut_safe,
                gamut_mode=args.gamut_mode,
                tone_rolloff=args.tone_rolloff,
                shadow_floor_l=args.shadow_floor_l,
                highlight_ceiling_l=args.highlight_ceiling_l,
                preserve_luma_detail_strength=args.preserve_luma_detail,
                chroma_curve_strength=args.chroma_curve_strength,
                output_margin=args.output_margin,
                guardrails=guardrail_config,
                neutral_protect=args.neutral_protect,
                skin_protect=args.skin_protect,
                max_chroma_gain=args.max_chroma_gain,
                max_chroma_boost=args.max_chroma_boost,
                max_chroma_absolute=args.max_chroma_absolute,
                dither=args.dither,
            )
            if args.output and len(styles) == 1 and len(strengths) == 1:
                output_path = args.output
            else:
                suffix = ".png" if args.format == "png" else ".jpg"
                output_path = output_dir / f"{args.input.stem}_{style}_s{slug_float(strength)}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_output(out, output_path, args.format)
            failed_clip_gate, clip_stats = saved_output_has_new_clip(image, output_path, args.output_margin)
            if args.fail_on_clip and failed_clip_gate:
                raise RuntimeError(f"Output clipping gate failed for {output_path}: {clip_stats}")
            row = {
                "input": str(args.input.resolve()),
                "style": style,
                "strength": strength,
                "luma_strength": args.luma_strength,
                "grain": args.grain,
                "gamut_safe": args.gamut_safe,
                "gamut_mode": args.gamut_mode or ("source" if args.gamut_safe else "off"),
                "tone_rolloff": args.tone_rolloff,
                "shadow_floor_l": args.shadow_floor_l,
                "highlight_ceiling_l": args.highlight_ceiling_l,
                "preserve_luma_detail": args.preserve_luma_detail,
                "chroma_curve_strength": args.chroma_curve_strength,
                "output_margin": args.output_margin,
                "guardrails": args.use_guardrails,
                "output": str(output_path),
                "label": f"{style} s={strength}",
                **clip_stats,
            }
            rows.append(row)
            style_rows.append(row)
            print(output_path)
        if args.contact_sheet and len(style_rows) > 1:
            write_contact_sheet(style_rows, output_dir, style)

    if len(rows) > 1:
        csv_path = output_dir / f"{args.input.stem}_color_baseline_manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / f"{args.input.stem}_color_baseline_manifest.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
