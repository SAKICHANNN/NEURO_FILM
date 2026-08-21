#!/usr/bin/env python3
"""Integrated content-preserving film renderer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import (
    load_guardrail_config,
    load_profile_values,
    style_transfer,
)
from src.filmfx import (
    PhysicalHalationControls,
    build_physical_halation_layer,
    composite_layers,
    describe_physical_halation_controls,
    dust_scratch_layer,
    get_halation_preset,
    grain_residual_layer,
    halation_layer,
    layer_metrics,
    physical_halation_layer,
    resolve_physical_halation_controls,
)
from src.inference import (
    atomic_write_json,
    build_render_recipe,
    load_render_profile,
    render_resolved_safe_lab_rgb,
    sha256_file,
)
from src.inference.analytic_render_recipe import build_analytic_render_recipe
from src.inference.analytic_y_chromaticity_profile_v4 import (
    load_analytic_y_chromaticity_profile,
    render_analytic_y_chromaticity_profile,
)
from src.preprocess import (
    load_working_image,
    resolve_look_approximation_claim,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile_fingerprint_sha256,
    srgb_icc_profile_sha256,
    working_image_to_srgb_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a content-preserving film look."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--style", default="velvia_50")
    parser.add_argument(
        "--color-engine",
        choices=("safe_lab", "analytic-y-chromaticity"),
        default="safe_lab",
    )
    parser.add_argument("--preset", choices=("safe-rich",), default="safe-rich")
    parser.add_argument(
        "--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json"
    )
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=ROOT / "configs" / "color_rendering_profiles.yaml",
    )
    parser.add_argument(
        "--render-profile",
        type=Path,
        default=ROOT / "configs" / "render_profiles" / "safe_rich_v1.json",
    )
    parser.add_argument(
        "--analytic-profile",
        type=Path,
        default=ROOT
        / "configs"
        / "render_profiles"
        / "analytic_y_chromaticity_cb69_v4.json",
    )
    parser.add_argument(
        "--analytic-scratch-root",
        type=Path,
        default=None,
        help="Existing local directory for analytic selector scratch files.",
    )
    parser.add_argument(
        "--use-render-profile",
        action="store_true",
        help="Source colour parameters from the validated versioned render profile.",
    )
    parser.add_argument(
        "--guardrails", type=Path, default=ROOT / "configs" / "color_guardrails.json"
    )
    parser.add_argument("--grain", type=float, default=0.0)
    parser.add_argument("--halation", type=float, default=0.0)
    parser.add_argument(
        "--halation-model", choices=("simple", "physical"), default="simple"
    )
    parser.add_argument("--halation-preset", default=None)
    parser.add_argument(
        "--halation-model-family",
        choices=("auto", "color_negative_backscatter", "bw_density_halation"),
        default="auto",
    )
    parser.add_argument(
        "--halation-type",
        choices=(
            "auto",
            "vision3_ahu",
            "cinestill_no_remjet",
            "classic_dense_base",
            "bw_clear_base",
        ),
        default="auto",
    )
    parser.add_argument(
        "--halation-color-response",
        choices=(
            "red_orange_core",
            "deep_red",
            "amber_core",
            "neutral_density",
            "warm_neutral_density",
        ),
        default=None,
    )
    parser.add_argument(
        "--halation-profile",
        choices=("vision3_500t", "cinestill_800t", "generic"),
        default=None,
    )
    parser.add_argument(
        "--halation-control-mode", choices=("locked", "expert"), default="locked"
    )
    parser.add_argument(
        "--halation-physics-lock",
        dest="halation_control_mode",
        action="store_const",
        const="locked",
    )
    parser.add_argument(
        "--halation-expert-controls",
        dest="halation_control_mode",
        action="store_const",
        const="expert",
    )
    parser.add_argument("--halation-amount", type=float, default=None)
    parser.add_argument("--halation-impact", type=float, default=None)
    parser.add_argument("--halation-anti-halation", type=float, default=None)
    parser.add_argument("--halation-source-selectivity", type=float, default=None)
    parser.add_argument("--halation-diffusion", type=float, default=None)
    parser.add_argument("--halation-warm-core", type=float, default=None)
    parser.add_argument("--halation-background-visibility", type=float, default=None)
    parser.add_argument(
        "--halation-source-normalization",
        choices=("percentile", "none"),
        default="percentile",
    )
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
    parser.add_argument("--output-bit-depth", type=int, choices=(8, 16), default=8)
    parser.add_argument("--write-layers", action="store_true")
    parser.add_argument("--write-metrics", action="store_true")
    parser.add_argument("--write-recipe", action="store_true")
    return parser.parse_args()


def save_rgb(rgb: np.ndarray, path: Path, bit_depth: int = 8) -> str:
    if bit_depth == 8:
        return save_srgb8(rgb, path)
    if path.suffix.casefold() == ".png":
        return save_srgb16_png(rgb, path)
    if path.suffix.casefold() in {".tif", ".tiff"}:
        return save_srgb16_tiff(rgb, path)
    raise ValueError("16-bit output requires .png, .tif or .tiff")


def _arg_or(value, fallback):
    return fallback if value is None else value


def _verify_recipe_profile_assets(profile: dict, args: argparse.Namespace) -> None:
    expected = {
        "legacy_profile_config": args.profile_config.resolve(),
        "style_statistics": args.stats.resolve(),
        "color_guardrails": args.guardrails.resolve(),
    }
    for asset in profile["assets"]:
        role = asset["role"]
        path = (ROOT / asset["path"]).resolve()
        if expected.get(role) != path or sha256_file(expected[role]) != asset["sha256"]:
            raise ValueError(f"Recipe profile asset mismatch for {role}")


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


def build_color_render_float(
    rgb: np.ndarray,
    args: argparse.Namespace,
    *,
    profile_values: dict | None = None,
) -> np.ndarray:
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    profile = (
        load_profile_values(args.profile_config, args.preset, args.style)
        if profile_values is None
        else profile_values
    )
    return render_resolved_safe_lab_rgb(
        rgb,
        style=args.style,
        style_statistics=stats["styles"][args.style],
        style_parameters=profile,
        guardrails=load_guardrail_config(args.guardrails, args.style),
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()
    if args.output_bit_depth == 16 and args.output.suffix.casefold() not in {
        ".png",
        ".tif",
        ".tiff",
    }:
        raise ValueError("16-bit output requires .png, .tif or .tiff")
    analytic_runtime = None
    color_diagnostics = None
    if args.color_engine == "analytic-y-chromaticity":
        if args.style != "velvia_50":
            raise ValueError(
                "analytic Y/chromaticity research profile only supports velvia_50"
            )
        if args.use_render_profile:
            raise ValueError(
                "safe-Lab --use-render-profile cannot be combined with the analytic research engine"
            )
        analytic_runtime = load_analytic_y_chromaticity_profile(
            args.analytic_profile, root=ROOT
        )
        if (
            args.analytic_scratch_root is not None
            and not args.analytic_scratch_root.is_dir()
        ):
            raise ValueError("analytic scratch root must be an existing directory")
    profile_manifest = None
    profile_values = None
    if args.use_render_profile or (args.write_recipe and analytic_runtime is None):
        profile_manifest = load_render_profile(args.render_profile, root=ROOT)
        _verify_recipe_profile_assets(profile_manifest, args)
        if args.style not in profile_manifest["style_parameters"]:
            raise ValueError(f"Render profile does not contain style {args.style!r}")
        profile_values = dict(profile_manifest["style_parameters"][args.style])
        if not args.use_render_profile:
            legacy_values = load_profile_values(
                args.profile_config, args.preset, args.style
            )
            if profile_values != legacy_values:
                raise ValueError(
                    f"Recipe profile does not exactly migrate style {args.style!r}"
                )
    working = load_working_image(args.input)
    output_claim = resolve_look_approximation_claim(working)
    if analytic_runtime is None:
        base = build_color_render_float(
            working_image_to_srgb_float(working),
            args,
            profile_values=profile_values if args.use_render_profile else None,
        )
    else:
        base, color_diagnostics = render_analytic_y_chromaticity_profile(
            working,
            analytic_runtime,
            scratch_root=args.analytic_scratch_root,
        )
    layers = []
    halation_resolved = None
    halation_metadata = None
    halation_preset_id = None
    if args.grain > 0:
        layers.append(
            grain_residual_layer(
                base,
                strength=args.grain,
                seed=args.seed,
                color=args.style not in {"hp5", "tri_x_400"},
            )
        )
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
                    color_response=_arg_or(
                        args.halation_color_response, base_controls.color_response
                    ),
                    profile=_arg_or(args.halation_profile, base_controls.profile),
                    amount=args.halation
                    if args.halation_amount is None
                    else args.halation_amount,
                    impact=_arg_or(args.halation_impact, base_controls.impact),
                    anti_halation=_arg_or(
                        args.halation_anti_halation, base_controls.anti_halation
                    ),
                    source_selectivity=_arg_or(
                        args.halation_source_selectivity,
                        base_controls.source_selectivity,
                    ),
                    diffusion=_arg_or(args.halation_diffusion, base_controls.diffusion),
                    warm_core=_arg_or(args.halation_warm_core, base_controls.warm_core),
                    background_visibility=_arg_or(
                        args.halation_background_visibility,
                        base_controls.background_visibility,
                    ),
                    source_normalization=args.halation_source_normalization,
                )
                halation_resolved = resolve_physical_halation_controls(controls)
                halation_metadata = describe_physical_halation_controls(controls)
                layers.append(build_physical_halation_layer(base, controls))
            else:
                no_remjet = (
                    None if args.halation_no_remjet < 0 else args.halation_no_remjet
                )
                halation_resolved = {
                    "profile": _arg_or(args.halation_profile, "cinestill_800t"),
                    "source_normalization": args.halation_source_normalization,
                    "amplify": args.halation
                    if args.halation_amount is None
                    else args.halation_amount,
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
        layers.append(
            dust_scratch_layer(base.shape, strength=args.dust, seed=args.seed + 17)
        )
    out = composite_layers(
        base,
        layers,
        output_margin=0 if analytic_runtime is not None else 4,
    )
    output_format = save_rgb(out, args.output, args.output_bit_depth)
    recipe_path = None
    recipe_sha256 = None
    if args.write_recipe:
        input_metadata = {
            "color_state": working.source_transfer_state,
            "working_space": working.working_space,
            "source_profile_kind": working.source_profile.kind,
            "bit_depth": working.bit_depth_in,
            "warnings": [warning.__dict__ for warning in working.warnings],
        }
        effects = {
            "grain": {
                "strength": args.grain,
                "seed": args.seed,
                "color": args.style not in {"hp5", "tri_x_400"},
            },
            "halation": {
                "strength": args.halation,
                "model": args.halation_model,
                "preset": halation_preset_id,
                "control_mode": args.halation_control_mode,
                "resolved_parameters": halation_resolved,
            },
            "dust": {"strength": args.dust, "seed": args.seed + 17},
        }
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
        if analytic_runtime is None:
            assert profile_manifest is not None and profile_values is not None
            recipe = build_render_recipe(
                profile_path=args.render_profile,
                profile=profile_manifest,
                input_path=args.input,
                input_metadata={
                    **input_metadata,
                    "source_profile_fingerprint_sha256": None,
                },
                render_metadata={
                    "engine_id": "safe_lab_v1",
                    "preset": args.preset,
                    "style": args.style,
                    "seed": args.seed,
                    "color_parameters": profile_values,
                    "effects": effects,
                },
                output_path=args.output,
                output_format=output_format,
                output_bit_depth=args.output_bit_depth,
                output_icc_fingerprint_sha256=srgb_icc_profile_fingerprint_sha256(),
                output_claim=output_claim,
                software_commit=commit,
            )
        else:
            assert color_diagnostics is not None
            recipe = build_analytic_render_recipe(
                runtime=analytic_runtime,
                input_path=args.input,
                input_metadata={
                    "source_color_state": working.source_transfer_state,
                    "runtime_transfer_state": working.transfer_state,
                    "working_space": working.working_space,
                    "source_profile_kind": working.source_profile.kind,
                    "bit_depth": working.bit_depth_in,
                    "warnings": [warning.__dict__ for warning in working.warnings],
                },
                selector_facts=color_diagnostics,
                effects=effects,
                output_path=args.output,
                output_format=output_format,
                output_bit_depth=args.output_bit_depth,
                output_icc_fingerprint_sha256=srgb_icc_profile_fingerprint_sha256(),
                output_claim=output_claim,
                software_commit=commit,
            )
        recipe_path = args.output.with_suffix(".recipe.json")
        recipe_sha256 = atomic_write_json(recipe_path, recipe)

    if args.write_layers:
        layer_dir = args.output.parent / f"{args.output.stem}_layers"
        for layer in layers:
            if layer.mode == "residual":
                view = np.clip(0.5 + layer.residual * 8.0, 0.0, 1.0)
            else:
                view = layer_on_black(layer)
            save_rgb(view, layer_dir / f"{layer.name}.png")
            if (
                layer.name in {"halation", "physical_halation", "density_halation"}
                and layer.mode == "screen"
            ):
                save_rgb(
                    layer_on_black(layer), layer_dir / f"{layer.name}_on_black.png"
                )
                save_rgb(
                    layer_on_white(layer), layer_dir / f"{layer.name}_on_white.png"
                )
    if args.write_metrics:
        quantization_max = 65535 if args.output_bit_depth == 16 else 255
        arr = np.rint(out * quantization_max).astype(
            np.uint16 if args.output_bit_depth == 16 else np.uint8
        )
        metrics = {
            "input": str(args.input),
            "output": str(args.output),
            "style": args.style,
            "color_engine": args.color_engine,
            "preset": args.preset,
            "profile_driven_adapter": args.use_render_profile,
            "analytic_research_profile": (
                None
                if analytic_runtime is None
                else {
                    "profile_id": analytic_runtime.profile["profile_id"],
                    "profile_version": analytic_runtime.profile["profile_version"],
                    "profile_sha256": analytic_runtime.profile_sha256,
                    "product_default": False,
                    "selector_facts": color_diagnostics,
                }
            ),
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
                "legacy_8bit_adapter": False,
                "internal_color_precision": "float32",
            },
            "output_encode": {
                "format": output_format,
                "bit_depth": args.output_bit_depth,
                "transfer": "sRGB",
                "icc_profile": "embedded standard sRGB",
                "icc_profile_sha256": srgb_icc_profile_sha256(),
                "icc_profile_fingerprint_sha256": srgb_icc_profile_fingerprint_sha256(),
            },
            "bounds": [int(arr.min()), int(arr.max())],
            "layers": [layer_metrics(layer) for layer in layers],
            "halation_control_mode": args.halation_control_mode
            if halation_resolved
            else None,
            "halation_preset": halation_preset_id,
            "halation_metadata": halation_metadata,
            "halation_resolved": halation_resolved,
        }
        if recipe_path is not None:
            metrics["render_recipe"] = {
                "schema_id": recipe["schema_id"],
                "path": str(recipe_path),
                "sha256": recipe_sha256,
            }
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
