"""Physically locked user controls for halation rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MODEL_FAMILY_CHOICES = ("auto", "color_negative_backscatter", "bw_density_halation")
HALATION_TYPE_CHOICES = ("auto", "vision3_ahu", "cinestill_no_remjet", "classic_dense_base", "bw_clear_base")
COLOR_RESPONSE_CHOICES = ("red_orange_core", "deep_red", "amber_core", "neutral_density", "warm_neutral_density")
PROFILE_CHOICES = ("vision3_500t", "cinestill_800t", "generic")
COLOR_NEGATIVE_RESPONSES = ("red_orange_core", "deep_red", "amber_core")
BW_DENSITY_RESPONSES = ("neutral_density", "warm_neutral_density")
EVIDENCE_HEURISTIC = "uncalibrated_heuristic"


@dataclass(frozen=True)
class PhysicalHalationControls:
    """Small, linked control surface for physical-prior halation.

    The lock deliberately keeps user-facing sliders from changing unrelated
    physical terms. In particular, ``amount`` changes scattered exposure
    coupling, not kernel radius; geometric radius is controlled by ``diffusion``.
    """

    model_family: str = "auto"
    halation_type: str = "auto"
    color_response: str = "red_orange_core"
    profile: str = "cinestill_800t"
    amount: float = 1.0
    impact: float = 0.85
    anti_halation: float = 0.75
    source_selectivity: float = 0.45
    diffusion: float = 0.55
    warm_core: float = 0.45
    background_visibility: float = 0.75
    source_normalization: str = "percentile"


@dataclass(frozen=True)
class HalationPreset:
    """Named GUI/API preset for locked halation controls."""

    preset_id: str
    label: str
    description: str
    controls: PhysicalHalationControls
    evidence_level: str = EVIDENCE_HEURISTIC


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _lerp(low: float, high: float, value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return low + (high - low) * value


def _profile_defaults(profile: str) -> dict[str, float]:
    if profile == "vision3_500t":
        return {
            "anti_halation_min": 0.24,
            "anti_halation_max": 0.58,
            "diffusion_min": 0.72,
            "diffusion_max": 1.22,
            "global_min": 0.05,
            "global_max": 0.18,
            "green_min": 0.10,
            "green_max": 0.30,
            "softness": 0.44,
            "gamma": 1.48,
            "alpha_cap": 0.22,
        }
    if profile == "cinestill_800t":
        return {
            "anti_halation_min": 0.72,
            "anti_halation_max": 1.20,
            "diffusion_min": 0.92,
            "diffusion_max": 1.70,
            "global_min": 0.12,
            "global_max": 0.36,
            "green_min": 0.18,
            "green_max": 0.58,
            "softness": 0.42,
            "gamma": 1.45,
            "alpha_cap": 0.32,
        }
    if profile == "generic":
        return {
            "anti_halation_min": 0.40,
            "anti_halation_max": 0.95,
            "diffusion_min": 0.82,
            "diffusion_max": 1.46,
            "global_min": 0.08,
            "global_max": 0.28,
            "green_min": 0.14,
            "green_max": 0.42,
            "softness": 0.43,
            "gamma": 1.46,
            "alpha_cap": 0.28,
        }
    raise ValueError(f"Unsupported halation profile: {profile}")


def _infer_type_from_profile(profile: str) -> str:
    if profile == "vision3_500t":
        return "vision3_ahu"
    if profile == "cinestill_800t":
        return "cinestill_no_remjet"
    return "classic_dense_base"


def _type_defaults(halation_type: str, profile: str) -> dict[str, float | str | tuple[float, float, float]]:
    if halation_type == "auto":
        halation_type = _infer_type_from_profile(profile)
    if halation_type == "vision3_ahu":
        return {
            "model_family": "color_negative_backscatter",
            "profile": "vision3_500t",
            "anti_halation_min": 0.22,
            "anti_halation_max": 0.54,
            "diffusion_min": 0.68,
            "diffusion_max": 1.16,
            "global_min": 0.04,
            "global_max": 0.15,
            "green_min": 0.08,
            "green_max": 0.26,
            "source_min": 2.35,
            "source_max": 3.45,
            "softness": 0.44,
            "gamma": 1.48,
            "alpha_cap": 0.20,
        }
    if halation_type == "cinestill_no_remjet":
        return {
            "model_family": "color_negative_backscatter",
            "profile": "cinestill_800t",
            "anti_halation_min": 0.78,
            "anti_halation_max": 1.26,
            "diffusion_min": 0.95,
            "diffusion_max": 1.75,
            "global_min": 0.12,
            "global_max": 0.38,
            "green_min": 0.18,
            "green_max": 0.58,
            "source_min": 1.35,
            "source_max": 2.75,
            "softness": 0.42,
            "gamma": 1.45,
            "alpha_cap": 0.32,
        }
    if halation_type == "classic_dense_base":
        return {
            "model_family": "color_negative_backscatter",
            "profile": "generic",
            "anti_halation_min": 0.34,
            "anti_halation_max": 0.84,
            "diffusion_min": 1.05,
            "diffusion_max": 1.82,
            "global_min": 0.08,
            "global_max": 0.26,
            "green_min": 0.10,
            "green_max": 0.36,
            "source_min": 1.85,
            "source_max": 3.10,
            "softness": 0.50,
            "gamma": 1.36,
            "alpha_cap": 0.24,
        }
    if halation_type == "bw_clear_base":
        return {
            "model_family": "bw_density_halation",
            "profile": "generic",
            "anti_halation_min": 0.0,
            "anti_halation_max": 0.0,
            "diffusion_min": 0.90,
            "diffusion_max": 1.92,
            "global_min": 0.10,
            "global_max": 0.36,
            "green_min": 0.0,
            "green_max": 0.0,
            "source_min": 1.55,
            "source_max": 3.00,
            "softness": 0.46,
            "gamma": 1.35,
            "alpha_cap": 0.28,
            "density_tint": (1.0, 0.96, 0.86),
        }
    raise ValueError(f"Unsupported halation_type: {halation_type}")


def _valid_color_responses_for_family(family: str) -> tuple[str, ...]:
    if family == "bw_density_halation":
        return BW_DENSITY_RESPONSES
    if family == "color_negative_backscatter":
        return COLOR_NEGATIVE_RESPONSES
    raise ValueError(f"Unsupported model_family: {family}")


def _resolve_type_and_family(controls: PhysicalHalationControls) -> tuple[str, str, dict[str, float | str | tuple[float, float, float]]]:
    if controls.model_family not in MODEL_FAMILY_CHOICES:
        raise ValueError(f"Unsupported model_family: {controls.model_family}")
    if controls.halation_type not in HALATION_TYPE_CHOICES:
        raise ValueError(f"Unsupported halation_type: {controls.halation_type}")
    defaults = _type_defaults(controls.halation_type, controls.profile)
    family = str(defaults["model_family"]) if controls.model_family == "auto" else controls.model_family
    if family not in MODEL_FAMILY_CHOICES or family == "auto":
        raise ValueError(f"Unsupported resolved model_family: {family}")
    halation_type = controls.halation_type if controls.halation_type != "auto" else _infer_type_from_profile(controls.profile)
    return halation_type, family, defaults


def validate_physical_halation_controls(controls: PhysicalHalationControls, *, strict: bool = True) -> None:
    """Validate locked halation controls.

    Strict mode matches the GUI/product contract. Compatibility mode keeps
    historical resolver fallback behavior for research use.
    """

    halation_type, family, _ = _resolve_type_and_family(controls)
    if controls.source_normalization not in {"percentile", "none"}:
        raise ValueError(f"Unsupported source_normalization: {controls.source_normalization}")
    if not strict:
        return
    valid_responses = _valid_color_responses_for_family(family)
    if controls.color_response not in valid_responses:
        raise ValueError(
            f"color_response={controls.color_response!r} is invalid for "
            f"halation_type={halation_type!r} / model_family={family!r}; "
            f"valid responses are {valid_responses}"
        )


def _color_response_modifiers(color_response: str, family: str) -> dict[str, float | tuple[float, float, float]]:
    if color_response not in COLOR_RESPONSE_CHOICES:
        raise ValueError(f"Unsupported color_response: {color_response}")
    if family == "bw_density_halation":
        if color_response == "neutral_density":
            return {"density_tint": (1.0, 1.0, 1.0)}
        if color_response == "warm_neutral_density":
            return {"density_tint": (1.0, 0.96, 0.86)}
        return {"density_tint": (1.0, 0.98, 0.92)}
    if color_response == "deep_red":
        return {"green_scale": 0.45}
    if color_response == "amber_core":
        return {"green_scale": 1.35}
    if color_response in {"neutral_density", "warm_neutral_density"}:
        raise ValueError(f"{color_response} is only valid for bw_density_halation")
    return {"green_scale": 1.0}


def describe_physical_halation_controls(
    controls: PhysicalHalationControls, *, strict: bool = True
) -> dict[str, float | str | tuple[float, float, float]]:
    validate_physical_halation_controls(controls, strict=strict)
    halation_type, family, defaults = _resolve_type_and_family(controls)
    return {
        "model_family": family,
        "halation_type": halation_type,
        "color_response": controls.color_response,
        "profile": str(defaults["profile"]),
    }


def resolve_physical_halation_controls(
    controls: PhysicalHalationControls, *, strict: bool = True
) -> dict[str, float | str | tuple[float, float, float]]:
    """Resolve locked user controls to ``physical_halation_layer`` kwargs.

    Locked invariants:
    - ``amount`` maps only to ``amplify``.
    - ``impact`` maps only to display ``impact``.
    - ``anti_halation`` maps only to backscatter coupling (``no_remjet``).
    - ``diffusion`` maps only to local/global diffusion radii.
    - ``source_selectivity`` maps only to source limiter threshold.
    - ``background_visibility`` maps only to dark-background gating.
    """

    validate_physical_halation_controls(controls, strict=strict)
    _, family, defaults = _resolve_type_and_family(controls)
    color_mod = _color_response_modifiers(controls.color_response, family)
    amount = _clamp(controls.amount, 0.0, 2.4)
    impact = _clamp(controls.impact, 0.0, 1.0)
    anti_halation = _clamp(controls.anti_halation, 0.0, 1.0)
    source_selectivity = _clamp(controls.source_selectivity, 0.0, 1.0)
    diffusion = _clamp(controls.diffusion, 0.0, 1.0)
    warm_core = _clamp(controls.warm_core, 0.0, 1.0)
    background_visibility = _clamp(controls.background_visibility, 0.0, 1.0)

    if controls.source_normalization not in {"percentile", "none"}:
        raise ValueError(f"Unsupported source_normalization: {controls.source_normalization}")

    local_diffusion = _lerp(defaults["diffusion_min"], defaults["diffusion_max"], diffusion)
    global_diffusion = _lerp(defaults["global_min"], defaults["global_max"], diffusion)
    source_limiter_stops = _lerp(defaults["source_min"], defaults["source_max"], source_selectivity)
    if family == "bw_density_halation":
        density_tint = color_mod.get("density_tint", defaults.get("density_tint", (1.0, 0.96, 0.86)))
        return {
            "source_normalization": controls.source_normalization,
            "amplify": amount,
            "impact": impact,
            "source_limiter_stops": source_limiter_stops,
            "source_softness": float(defaults["softness"]),
            "source_gamma": float(defaults["gamma"]),
            "local_diffusion": local_diffusion,
            "global_diffusion": global_diffusion,
            "background_gain": _lerp(0.85, 1.85, background_visibility),
            "background_luma_target": _lerp(0.14, 0.24, background_visibility),
            "density_tint": density_tint,
            "output_alpha_cap": float(defaults["alpha_cap"]),
        }

    green_scale = float(color_mod.get("green_scale", 1.0))
    green_min = float(defaults["green_min"]) * green_scale
    green_max = float(defaults["green_max"]) * green_scale

    return {
        "profile": str(defaults["profile"]),
        "source_normalization": controls.source_normalization,
        "amplify": amount,
        "impact": impact,
        "source_limiter_stops": source_limiter_stops,
        "source_softness": float(defaults["softness"]),
        "source_gamma": float(defaults["gamma"]),
        "local_diffusion": local_diffusion,
        "global_diffusion": global_diffusion,
        "hue_green": min(_lerp(green_min, green_max, warm_core), 0.72),
        "background_gain": _lerp(0.85, 1.85, background_visibility),
        "background_luma_target": _lerp(0.14, 0.24, background_visibility),
        "no_remjet": _lerp(defaults["anti_halation_min"], defaults["anti_halation_max"], anti_halation),
        "output_alpha_cap": float(defaults["alpha_cap"]),
    }


HALATION_PRESETS = {
    "vision3_clean": HalationPreset(
        preset_id="vision3_clean",
        label="Vision3 Clean",
        description="Restrained Vision3/AHU-like red-orange edge behavior.",
        controls=PhysicalHalationControls(
            halation_type="vision3_ahu",
            color_response="red_orange_core",
            profile="vision3_500t",
            amount=0.65,
            impact=0.72,
            anti_halation=0.10,
            source_selectivity=0.76,
            diffusion=0.30,
            warm_core=0.18,
            background_visibility=0.64,
        ),
    ),
    "vision3_push": HalationPreset(
        preset_id="vision3_push",
        label="Vision3 Push",
        description="Still restrained, but with a more visible pushed highlight edge.",
        controls=PhysicalHalationControls(
            halation_type="vision3_ahu",
            color_response="red_orange_core",
            profile="vision3_500t",
            amount=0.95,
            impact=0.86,
            anti_halation=0.32,
            source_selectivity=0.62,
            diffusion=0.46,
            warm_core=0.30,
            background_visibility=0.70,
        ),
    ),
    "cinestill_balanced": HalationPreset(
        preset_id="cinestill_balanced",
        label="CineStill Balanced",
        description="No-remjet-like red/orange halation with balanced visibility.",
        controls=PhysicalHalationControls(
            halation_type="cinestill_no_remjet",
            color_response="red_orange_core",
            profile="cinestill_800t",
            amount=1.08,
            impact=0.86,
            anti_halation=0.78,
            source_selectivity=0.48,
            diffusion=0.55,
            warm_core=0.45,
            background_visibility=0.78,
        ),
    ),
    "cinestill_strong": HalationPreset(
        preset_id="cinestill_strong",
        label="CineStill Strong",
        description="More assertive no-remjet-like halation for high-impact previews.",
        controls=PhysicalHalationControls(
            halation_type="cinestill_no_remjet",
            color_response="red_orange_core",
            profile="cinestill_800t",
            amount=1.40,
            impact=0.96,
            anti_halation=1.00,
            source_selectivity=0.36,
            diffusion=0.70,
            warm_core=0.60,
            background_visibility=0.86,
        ),
    ),
    "cinestill_amber": HalationPreset(
        preset_id="cinestill_amber",
        label="CineStill Amber",
        description="No-remjet-like geometry with stronger amber core response.",
        controls=PhysicalHalationControls(
            halation_type="cinestill_no_remjet",
            color_response="amber_core",
            profile="cinestill_800t",
            amount=1.20,
            impact=0.88,
            anti_halation=0.84,
            source_selectivity=0.46,
            diffusion=0.56,
            warm_core=0.58,
            background_visibility=0.78,
        ),
    ),
    "classic_soft": HalationPreset(
        preset_id="classic_soft",
        label="Classic Soft",
        description="Soft dense-base-like color negative glow.",
        controls=PhysicalHalationControls(
            halation_type="classic_dense_base",
            color_response="red_orange_core",
            profile="generic",
            amount=1.05,
            impact=0.82,
            anti_halation=0.52,
            source_selectivity=0.56,
            diffusion=0.68,
            warm_core=0.30,
            background_visibility=0.70,
        ),
    ),
    "bw_neutral": HalationPreset(
        preset_id="bw_neutral",
        label="B&W Neutral",
        description="Neutral density-family halation for monochrome-like looks.",
        controls=PhysicalHalationControls(
            halation_type="bw_clear_base",
            color_response="neutral_density",
            profile="generic",
            amount=1.18,
            impact=0.84,
            anti_halation=0.0,
            source_selectivity=0.50,
            diffusion=0.66,
            warm_core=0.0,
            background_visibility=0.76,
        ),
    ),
    "bw_warm": HalationPreset(
        preset_id="bw_warm",
        label="B&W Warm",
        description="Warm-neutral density-family halation.",
        controls=PhysicalHalationControls(
            halation_type="bw_clear_base",
            color_response="warm_neutral_density",
            profile="generic",
            amount=1.18,
            impact=0.84,
            anti_halation=0.0,
            source_selectivity=0.50,
            diffusion=0.66,
            warm_core=0.0,
            background_visibility=0.76,
        ),
    ),
}


def get_halation_preset(preset_id: str) -> HalationPreset:
    try:
        return HALATION_PRESETS[preset_id]
    except KeyError as exc:
        choices = tuple(HALATION_PRESETS)
        raise ValueError(f"Unsupported halation preset: {preset_id}; choices are {choices}") from exc


def list_halation_presets() -> list[HalationPreset]:
    return list(HALATION_PRESETS.values())


def halation_gui_schema() -> dict:
    """Return a machine-readable GUI contract for locked halation controls."""

    sliders = {
        "amount": {"label": "Amount", "min": 0.0, "max": 2.4, "step": 0.01, "default": 1.0},
        "impact": {"label": "Impact", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.85},
        "anti_halation": {"label": "Anti-Halation Loss", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.75},
        "source_selectivity": {"label": "Source Selectivity", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.45},
        "diffusion": {"label": "Diffusion", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.55},
        "warm_core": {"label": "Warm Core", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.45},
        "background_visibility": {
            "label": "Background Visibility",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.75,
        },
    }
    types = []
    for halation_type in HALATION_TYPE_CHOICES:
        if halation_type == "auto":
            continue
        defaults = _type_defaults(halation_type, "generic")
        family = str(defaults["model_family"])
        types.append(
            {
                "id": halation_type,
                "model_family": family,
                "valid_color_responses": list(_valid_color_responses_for_family(family)),
                "show_sliders": [
                    name
                    for name in sliders
                    if family == "color_negative_backscatter" or name not in {"anti_halation", "warm_core"}
                ],
                "evidence_level": EVIDENCE_HEURISTIC,
            }
        )
    return {
        "version": "halation_v2p3",
        "model_families": [choice for choice in MODEL_FAMILY_CHOICES if choice != "auto"],
        "types": types,
        "color_responses": list(COLOR_RESPONSE_CHOICES),
        "sliders": sliders,
        "preview_modes": ["combined", "layer_black", "layer_white", "original"],
        "presets": [
            {
                "id": preset.preset_id,
                "label": preset.label,
                "description": preset.description,
                "controls": asdict(preset.controls),
                "metadata": describe_physical_halation_controls(preset.controls),
                "evidence_level": preset.evidence_level,
            }
            for preset in list_halation_presets()
        ],
        "evidence_levels": {
            "presets": EVIDENCE_HEURISTIC,
            "slider_ranges": EVIDENCE_HEURISTIC,
            "valid_combinations": "code_fact",
        },
    }


def build_physical_halation_layer(base_rgb: np.ndarray, controls: PhysicalHalationControls, *, strict: bool = True):
    """Build a halation layer from locked controls and its resolved rule family."""

    _, family, _ = _resolve_type_and_family(controls)
    resolved = resolve_physical_halation_controls(controls, strict=strict)
    if family == "bw_density_halation":
        from .effects import density_halation_layer

        return density_halation_layer(base_rgb, **resolved)
    from .effects import physical_halation_layer

    return physical_halation_layer(base_rgb, **resolved)
