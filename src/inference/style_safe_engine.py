"""Pure deterministic safe-Lab rendering entry points."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import (
    load_guardrail_config,
    style_transfer_rgb,
    style_transfer_rgb_tiled,
)
from src.filmfx import (
    composite_layers,
    density_halation_layer,
    dust_scratch_layer,
    grain_residual_layer,
    halation_layer,
    physical_halation_layer,
)
from src.preprocess import (
    WorkingImage,
    load_working_image,
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
    srgb_icc_profile_fingerprint_sha256,
    working_image_to_srgb_float,
)

from .render_contract import (
    COLOR_PARAMETER_KEYS,
    load_render_profile,
    sha256_file,
    validate_render_profile,
    verify_render_recipe_inputs,
)


class StyleSafeEngineError(ValueError):
    """Raised when a resolved style or engine result violates the v1 contract."""


def _validated_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if set(parameters) != COLOR_PARAMETER_KEYS:
        raise StyleSafeEngineError("safe-Lab style parameter keys drifted")
    numeric = COLOR_PARAMETER_KEYS - {
        "gamut_safe",
        "gamut_mode",
        "use_guardrails",
    }
    if any(
        isinstance(parameters[key], bool)
        or not isinstance(parameters[key], (int, float))
        or not math.isfinite(float(parameters[key]))
        for key in numeric
    ):
        raise StyleSafeEngineError("safe-Lab style parameters must be finite")
    if not isinstance(parameters["gamut_safe"], bool) or not isinstance(
        parameters["use_guardrails"], bool
    ):
        raise StyleSafeEngineError("safe-Lab boolean parameters drifted")
    if parameters["gamut_mode"] not in {"off", "source", "chroma"}:
        raise StyleSafeEngineError("safe-Lab gamut mode drifted")
    return dict(parameters)


def render_resolved_safe_lab_rgb(
    encoded_srgb: np.ndarray,
    *,
    style: str,
    style_statistics: Mapping[str, Any],
    style_parameters: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
    tile_size: int | None = None,
) -> np.ndarray:
    """Render one already-resolved style without file or CLI state."""

    source = np.asarray(encoded_srgb)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or source.size == 0
        or not np.isfinite(source).all()
        or np.any((source < 0.0) | (source > 1.0))
    ):
        raise StyleSafeEngineError("encoded_srgb must be finite bounded HxWx3 float32")
    if not isinstance(style, str) or not style:
        raise StyleSafeEngineError("style must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StyleSafeEngineError("seed must be an integer")
    if tile_size is not None and (
        isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1
    ):
        raise StyleSafeEngineError("tile_size must be a positive integer")
    parameters = _validated_parameters(style_parameters)
    if tile_size is not None:
        output, _ = style_transfer_rgb_tiled(
            source,
            style_statistics,
            style,
            strength=float(parameters["strength"]),
            luma_strength=float(parameters["luma_strength"]),
            grain=float(parameters["grain"]),
            seed=seed,
            gamut_safe=parameters["gamut_safe"],
            gamut_mode=parameters["gamut_mode"],
            tone_rolloff=float(parameters["tone_rolloff"]),
            shadow_floor_l=float(parameters["shadow_floor_l"]),
            highlight_ceiling_l=float(parameters["highlight_ceiling_l"]),
            preserve_luma_detail_strength=float(parameters["preserve_luma_detail"]),
            chroma_curve_strength=float(parameters["chroma_curve_strength"]),
            output_margin=int(parameters["output_margin"]),
            guardrails=dict(guardrails),
            dither=float(parameters["dither"]),
            tile_size=tile_size,
        )
        return np.ascontiguousarray(output, dtype=np.float32)
    output = np.ascontiguousarray(
        style_transfer_rgb(
            source,
            dict(style_statistics),
            style,
            strength=float(parameters["strength"]),
            luma_strength=float(parameters["luma_strength"]),
            grain=float(parameters["grain"]),
            seed=seed,
            gamut_safe=parameters["gamut_safe"],
            gamut_mode=parameters["gamut_mode"],
            tone_rolloff=float(parameters["tone_rolloff"]),
            shadow_floor_l=float(parameters["shadow_floor_l"]),
            highlight_ceiling_l=float(parameters["highlight_ceiling_l"]),
            preserve_luma_detail_strength=float(
                parameters["preserve_luma_detail"]
            ),
            chroma_curve_strength=float(parameters["chroma_curve_strength"]),
            output_margin=int(parameters["output_margin"]),
            guardrails=dict(guardrails) if parameters["use_guardrails"] else None,
            dither=float(parameters["dither"]),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise StyleSafeEngineError("safe-Lab output is non-finite or unbounded")
    return output


def render_style_safe_working_image(
    working: WorkingImage,
    *,
    profile: Mapping[str, Any],
    style: str,
    style_statistics: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
    tile_size: int | None = None,
) -> np.ndarray:
    """Render a validated v1 profile from one WorkingImage to float32 sRGB."""

    validate_render_profile(profile)
    styles = profile["style_parameters"]
    if style not in styles:
        raise StyleSafeEngineError(f"style is absent from profile: {style}")
    source = working_image_to_srgb_float(working)
    return render_resolved_safe_lab_rgb(
        source,
        style=style,
        style_statistics=style_statistics,
        style_parameters=styles[style],
        guardrails=guardrails,
        seed=seed,
        tile_size=tile_size,
    )


def _verified_recipe_base(
    recipe: Mapping[str, Any],
    *,
    profile_path: Path,
    root: Path,
    tile_size: int | None = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:

    verify_render_recipe_inputs(recipe, profile_path=profile_path, root=root)
    profile = load_render_profile(profile_path, root=root)
    render = recipe["render"]
    style = render["style"]
    if style not in profile["style_parameters"]:
        raise StyleSafeEngineError("recipe style is absent from profile")
    if render["color_parameters"] != profile["style_parameters"][style]:
        raise StyleSafeEngineError("recipe color parameters differ from profile")
    assets = {asset["role"]: root / asset["path"] for asset in recipe["assets"]}
    statistics = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
    if style not in statistics.get("styles", {}):
        raise StyleSafeEngineError("recipe style statistics are absent")
    input_metadata = recipe["input"]
    if input_metadata["source_profile_fingerprint_sha256"] is not None:
        raise StyleSafeEngineError("v1 replay cannot verify source profile fingerprint")
    working = load_working_image(Path(input_metadata["path"]))
    actual_metadata = {
        "color_state": working.source_transfer_state,
        "working_space": working.working_space,
        "source_profile_kind": working.source_profile.kind,
        "bit_depth": working.bit_depth_in,
        "warnings": [warning.__dict__ for warning in working.warnings],
    }
    expected_metadata = {
        key: input_metadata[key]
        for key in (
            "color_state",
            "working_space",
            "source_profile_kind",
            "bit_depth",
            "warnings",
        )
    }
    if actual_metadata != expected_metadata:
        raise StyleSafeEngineError("recipe decoded input metadata drifted")
    base = render_style_safe_working_image(
        working,
        profile=profile,
        style=style,
        style_statistics=statistics["styles"][style],
        guardrails=load_guardrail_config(assets["color_guardrails"], style),
        seed=render["seed"],
        tile_size=tile_size,
    )
    return base, render


def replay_style_safe_color_recipe(
    recipe: Mapping[str, Any],
    *,
    profile_path: Path,
    root: Path,
    tile_size: int | None = None,
) -> np.ndarray:
    """Verify and replay the color-only stage of one existing v1 recipe."""

    base, render = _verified_recipe_base(
        recipe, profile_path=profile_path, root=root, tile_size=tile_size
    )
    effects = render["effects"]
    if any(
        float(effects[name]["strength"]) != 0.0
        for name in ("grain", "halation", "dust")
    ):
        raise StyleSafeEngineError("color-only replay rejects enabled effects")
    return base


def replay_style_safe_recipe(
    recipe: Mapping[str, Any],
    *,
    profile_path: Path,
    root: Path,
    tile_size: int | None = None,
) -> np.ndarray:
    """Verify and replay all deterministic v1 safe-Lab recipe stages."""

    base, render = _verified_recipe_base(
        recipe, profile_path=profile_path, root=root, tile_size=tile_size
    )
    effects = render["effects"]
    layers = []
    grain = effects["grain"]
    if float(grain["strength"]) > 0.0:
        layers.append(
            grain_residual_layer(
                base,
                strength=float(grain["strength"]),
                seed=int(grain["seed"]),
                color=grain["color"],
            )
        )
    halation = effects["halation"]
    if float(halation["strength"]) > 0.0:
        if halation["model"] == "simple":
            layers.append(halation_layer(base, strength=float(halation["strength"])))
        else:
            resolved = halation["resolved_parameters"]
            if not isinstance(resolved, Mapping) or not resolved:
                raise StyleSafeEngineError(
                    "physical halation recipe lacks resolved parameters"
                )
            kwargs = dict(resolved)
            if "density_tint" in kwargs:
                layers.append(density_halation_layer(base, **kwargs))
            else:
                layers.append(physical_halation_layer(base, **kwargs))
    dust = effects["dust"]
    if float(dust["strength"]) > 0.0:
        layers.append(
            dust_scratch_layer(
                base.shape,
                strength=float(dust["strength"]),
                seed=int(dust["seed"]),
            )
        )
    return composite_layers(base, layers, output_margin=4)


def replay_style_safe_recipe_to_file(
    recipe: Mapping[str, Any],
    *,
    profile_path: Path,
    output_path: Path,
    root: Path,
    tile_size: int | None = None,
) -> str:
    """Replay one v1 recipe to a new file and require the original byte identity."""

    if output_path.exists():
        raise StyleSafeEngineError("replay output path already exists")
    output = recipe.get("output")
    if not isinstance(output, Mapping):
        raise StyleSafeEngineError("recipe output is invalid")
    if (
        output.get("icc_profile_fingerprint_sha256")
        != srgb_icc_profile_fingerprint_sha256()
    ):
        raise StyleSafeEngineError("recipe output ICC fingerprint is unsupported")
    rendered = replay_style_safe_recipe(
        recipe, profile_path=profile_path, root=root, tile_size=tile_size
    )
    bit_depth = output.get("bit_depth")
    format_name = output.get("format")
    suffix = output_path.suffix.casefold()
    try:
        if bit_depth == 8 and format_name in {"PNG", "JPEG", "TIFF"}:
            actual_format = save_srgb8(rendered, output_path)
        elif bit_depth == 16 and format_name == "PNG" and suffix == ".png":
            actual_format = save_srgb16_png(rendered, output_path)
        elif bit_depth == 16 and format_name == "TIFF" and suffix in {".tif", ".tiff"}:
            actual_format = save_srgb16_tiff(rendered, output_path)
        else:
            raise StyleSafeEngineError("recipe output encoding is unsupported")
        if actual_format != format_name:
            raise StyleSafeEngineError("replay output extension differs from recipe format")
        digest = sha256_file(output_path)
        if digest != output.get("sha256"):
            raise StyleSafeEngineError("replayed output byte identity differs from recipe")
        return digest
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
