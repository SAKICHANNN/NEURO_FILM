#!/usr/bin/env python3
"""Integrated content-preserving film renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.filmfx import (  # noqa: E402
    composite_layers,
    dust_scratch_layer,
    grain_residual_layer,
    halation_layer,
    layer_metrics,
    physical_halation_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a content-preserving film look.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--style", default="velvia_50")
    parser.add_argument("--color-engine", choices=("safe_lab",), default="safe_lab")
    parser.add_argument("--preset", choices=("safe-rich",), default="safe-rich")
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    parser.add_argument("--profile-config", type=Path, default=ROOT / "configs" / "color_rendering_profiles.yaml")
    parser.add_argument("--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json")
    parser.add_argument("--grain", type=float, default=0.0)
    parser.add_argument("--halation", type=float, default=0.0)
    parser.add_argument("--halation-model", choices=("simple", "physical"), default="simple")
    parser.add_argument("--halation-profile", choices=("vision3_500t", "cinestill_800t", "generic"), default="cinestill_800t")
    parser.add_argument("--halation-impact", type=float, default=0.85)
    parser.add_argument("--halation-source-limiter", type=float, default=2.0)
    parser.add_argument("--halation-local-diffusion", type=float, default=1.0)
    parser.add_argument("--halation-global-diffusion", type=float, default=0.18)
    parser.add_argument("--halation-hue-green", type=float, default=0.28)
    parser.add_argument("--halation-background-gain", type=float, default=1.25)
    parser.add_argument("--halation-no-remjet", type=float, default=-1.0)
    parser.add_argument("--dust", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-layers", action="store_true")
    parser.add_argument("--write-metrics", action="store_true")
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path, "PNG")


def build_color_render(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    profile = load_profile_values(args.profile_config, args.preset, args.style)
    return style_transfer(
        image,
        stats["styles"][args.style],
        args.style,
        strength=profile["strength"],
        luma_strength=profile["luma_strength"],
        grain=profile["grain"],
        seed=args.seed,
        gamut_safe=profile["gamut_safe"],
        gamut_mode=profile["gamut_mode"],
        tone_rolloff=profile["tone_rolloff"],
        shadow_floor_l=profile["shadow_floor_l"],
        highlight_ceiling_l=profile["highlight_ceiling_l"],
        preserve_luma_detail_strength=profile["preserve_luma_detail"],
        chroma_curve_strength=profile["chroma_curve_strength"],
        output_margin=profile["output_margin"],
        guardrails=load_guardrail_config(args.guardrails, args.style),
        dither=profile["dither"],
    )


def main() -> int:
    args = parse_args()
    image = load_image(args.input)
    color_image = build_color_render(image, args)
    base = np.asarray(color_image, dtype=np.float32) / 255.0
    layers = []
    if args.grain > 0:
        layers.append(grain_residual_layer(base, strength=args.grain, seed=args.seed, color=args.style not in {"hp5", "tri_x_400"}))
    if args.halation > 0:
        if args.halation_model == "physical":
            no_remjet = None if args.halation_no_remjet < 0 else args.halation_no_remjet
            layers.append(
                physical_halation_layer(
                    base,
                    profile=args.halation_profile,
                    amplify=args.halation,
                    impact=args.halation_impact,
                    source_limiter_stops=args.halation_source_limiter,
                    local_diffusion=args.halation_local_diffusion,
                    global_diffusion=args.halation_global_diffusion,
                    hue_green=args.halation_hue_green,
                    background_gain=args.halation_background_gain,
                    no_remjet=no_remjet,
                )
            )
        else:
            layers.append(halation_layer(base, strength=args.halation))
    if args.dust > 0:
        layers.append(dust_scratch_layer(base.shape, strength=args.dust, seed=args.seed + 17))
    out = composite_layers(base, layers, output_margin=4)
    save_rgb(out, args.output)

    if args.write_layers:
        layer_dir = args.output.parent / f"{args.output.stem}_layers"
        for layer in layers:
            if layer.mode == "residual":
                view = np.clip(0.5 + layer.residual * 8.0, 0.0, 1.0)
            else:
                view = layer.rgb * layer.alpha
            save_rgb(view, layer_dir / f"{layer.name}.png")
    if args.write_metrics:
        arr = np.rint(out * 255.0).astype(np.uint8)
        metrics = {
            "input": str(args.input),
            "output": str(args.output),
            "style": args.style,
            "color_engine": args.color_engine,
            "preset": args.preset,
            "bounds": [int(arr.min()), int(arr.max())],
            "layers": [layer_metrics(layer) for layer in layers],
        }
        args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
