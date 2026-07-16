#!/usr/bin/env python3
"""Integrated content-preserving film renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values, style_transfer  # noqa: E402
from src.preprocess import (  # noqa: E402
    load_working_image,
    resolve_look_approximation_claim,
    save_srgb8,
    srgb_icc_profile_fingerprint_sha256,
    srgb_icc_profile_sha256,
    working_image_to_legacy_srgb8,
)
from src.filmfx import (  # noqa: E402
    PhysicalHalationControls,
    build_physical_halation_layer,
    composite_layers,
    describe_physical_halation_controls,
    dust_scratch_layer,
    grain_residual_layer,
    halation_layer,
    layer_metrics,
    physical_halation_layer,
    get_halation_preset,
    resolve_physical_halation_controls,
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
    parser.add_argument("--halation-preset", default=None)
    parser.add_argument("--halation-model-family", choices=("auto", "color_negative_backscatter", "bw_density_halation"), default="auto")
    parser.add_argument(
        "--halation-type",
        choices=("auto", "vision3_ahu", "cinestill_no_remjet", "classic_dense_base", "bw_clear_base"),
        default="auto",
    )
    parser.add_argument(
        "--halation-color-response",
        choices=("red_orange_core", "deep_red", "amber_core", "neutral_density", "warm_neutral_density"),
        default=None,
    )
    parser.add_argument("--halation-profile", choices=("vision3_500t", "cinestill_800t", "generic"), default=None)
    parser.add_argument("--halation-control-mode", choices=("locked", "expert"), default="locked")
    parser.add_argument("--halation-physics-lock", dest="halation_control_mode", action="store_const", const="locked")
    parser.add_argument("--halation-expert-controls", dest="halation_control_mode", action="store_const", const="expert")
    parser.add_argument("--halation-amount", type=float, default=None)
    parser.add_argument("--halation-impact", type=float, default=None)
    parser.add_argument("--halation-anti-halation", type=float, default=None)
    parser.add_argument("--halation-source-selectivity", type=float, default=None)
    parser.add_argument("--halation-diffusion", type=float, default=None)
    parser.add_argument("--halation-warm-core", type=float, default=None)
    parser.add_argument("--halation-background-visibility", type=float, default=None)
    parser.add_argument("--halation-source-normalization", choices=("percentile", "none"), default="percentile")
    parser.add_argument("--halation-source-limiter", type=float, default=2.0)
    parser.add_argument("--halation-local-diffusion", type=float, default=1.0)
    parser.add_argument("--halation-global-diffusion", type=float, default=0.18)
    parser.add_argument("--halation-hue-green", type=float, default=0.28)
    parser.add_argument("--halation-background-gain", type=float, default=1.25)
    parser.add_argument("--halation-background-luma-target", type=float, default=0.20)
    parser.add_argument("--halation-no-remjet", type=float, default=-1.0)
    parser.add_argument("--dust", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-layers", action="store_true")
    parser.add_argument("--write-metrics", action="store_true")
    return parser.parse_args()


def save_rgb(rgb: np.ndarray, path: Path) -> str:
    return save_srgb8(rgb, path)


def _arg_or(value, fallback):
    return fallback if value is None else value


def layer_on_black(layer) -> np.ndarray:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(layer.rgb * alpha, 0.0, 1.0)


def layer_on_white(layer) -> np.ndarray:
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(1.0 * (1.0 - alpha) + layer.rgb * alpha, 0.0, 1.0)


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
    working = load_working_image(args.input)
    output_claim = resolve_look_approximation_claim(working)
    image = working_image_to_legacy_srgb8(working)
    color_image = build_color_render(image, args)
    base = np.asarray(color_image, dtype=np.float32) / 255.0
    layers = []
    halation_resolved = None
    halation_metadata = None
    halation_preset_id = None
    if args.grain > 0:
        layers.append(grain_residual_layer(base, strength=args.grain, seed=args.seed, color=args.style not in {"hp5", "tri_x_400"}))
    if args.halation > 0:
        if args.halation_model == "physical":
            if args.halation_control_mode == "locked":
                base_controls = (
                    get_halation_preset(args.halation_preset).controls
                    if args.halation_preset
                    else PhysicalHalationControls()
                )
                halation_preset_id = args.halation_preset
                controls = PhysicalHalationControls(
                    model_family=base_controls.model_family
                    if args.halation_model_family == "auto"
                    else args.halation_model_family,
                    halation_type=base_controls.halation_type
                    if args.halation_type == "auto"
                    else args.halation_type,
                    color_response=_arg_or(args.halation_color_response, base_controls.color_response),
                    profile=_arg_or(args.halation_profile, base_controls.profile),
                    amount=args.halation if args.halation_amount is None else args.halation_amount,
                    impact=_arg_or(args.halation_impact, base_controls.impact),
                    anti_halation=_arg_or(args.halation_anti_halation, base_controls.anti_halation),
                    source_selectivity=_arg_or(args.halation_source_selectivity, base_controls.source_selectivity),
                    diffusion=_arg_or(args.halation_diffusion, base_controls.diffusion),
                    warm_core=_arg_or(args.halation_warm_core, base_controls.warm_core),
                    background_visibility=_arg_or(
                        args.halation_background_visibility, base_controls.background_visibility
                    ),
                    source_normalization=args.halation_source_normalization,
                )
                halation_resolved = resolve_physical_halation_controls(controls)
                halation_metadata = describe_physical_halation_controls(controls)
                layers.append(build_physical_halation_layer(base, controls))
            else:
                no_remjet = None if args.halation_no_remjet < 0 else args.halation_no_remjet
                halation_resolved = {
                    "profile": _arg_or(args.halation_profile, "cinestill_800t"),
                    "source_normalization": args.halation_source_normalization,
                    "amplify": args.halation if args.halation_amount is None else args.halation_amount,
                    "impact": _arg_or(args.halation_impact, 0.85),
                    "source_limiter_stops": args.halation_source_limiter,
                    "local_diffusion": args.halation_local_diffusion,
                    "global_diffusion": args.halation_global_diffusion,
                    "hue_green": args.halation_hue_green,
                    "background_gain": args.halation_background_gain,
                    "background_luma_target": args.halation_background_luma_target,
                    "no_remjet": no_remjet,
                }
                layers.append(
                    physical_halation_layer(
                        base,
                        **halation_resolved,
                    )
                )
        else:
            layers.append(halation_layer(base, strength=args.halation))
    if args.dust > 0:
        layers.append(dust_scratch_layer(base.shape, strength=args.dust, seed=args.seed + 17))
    out = composite_layers(base, layers, output_margin=4)
    output_format = save_rgb(out, args.output)

    if args.write_layers:
        layer_dir = args.output.parent / f"{args.output.stem}_layers"
        for layer in layers:
            if layer.mode == "residual":
                view = np.clip(0.5 + layer.residual * 8.0, 0.0, 1.0)
            else:
                view = layer_on_black(layer)
            save_rgb(view, layer_dir / f"{layer.name}.png")
            if layer.name in {"halation", "physical_halation", "density_halation"} and layer.mode == "screen":
                save_rgb(layer_on_black(layer), layer_dir / f"{layer.name}_on_black.png")
                save_rgb(layer_on_white(layer), layer_dir / f"{layer.name}_on_white.png")
    if args.write_metrics:
        arr = np.rint(out * 255.0).astype(np.uint8)
        metrics = {
            "input": str(args.input),
            "output": str(args.output),
            "style": args.style,
            "color_engine": args.color_engine,
            "preset": args.preset,
            "output_claim": output_claim,
            "input_decode": {
                "working_space": working.working_space,
                "transfer_state": working.transfer_state,
                "source_transfer_state": working.source_transfer_state,
                "source_profile_kind": working.source_profile.kind,
                "source_profile_description": working.source_profile.description,
                "bit_depth_in": working.bit_depth_in,
                "orientation_applied": working.orientation_applied,
                "alpha_policy": working.alpha_policy,
                "warnings": [warning.__dict__ for warning in working.warnings],
                "legacy_8bit_adapter": True,
            },
            "output_encode": {
                "format": output_format,
                "bit_depth": 8,
                "transfer": "sRGB",
                "icc_profile": "embedded standard sRGB",
                "icc_profile_sha256": srgb_icc_profile_sha256(),
                "icc_profile_fingerprint_sha256": srgb_icc_profile_fingerprint_sha256(),
            },
            "bounds": [int(arr.min()), int(arr.max())],
            "layers": [layer_metrics(layer) for layer in layers],
            "halation_control_mode": args.halation_control_mode if halation_resolved else None,
            "halation_preset": halation_preset_id,
            "halation_metadata": halation_metadata,
            "halation_resolved": halation_resolved,
        }
        args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
