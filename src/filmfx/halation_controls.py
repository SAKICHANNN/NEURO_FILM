"""Physically locked user controls for halation rendering."""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_CHOICES = ("vision3_500t", "cinestill_800t", "generic")


@dataclass(frozen=True)
class PhysicalHalationControls:
    """Small, linked control surface for physical-prior halation.

    The lock deliberately keeps user-facing sliders from changing unrelated
    physical terms. In particular, ``amount`` changes scattered exposure
    coupling, not kernel radius; geometric radius is controlled by ``diffusion``.
    """

    profile: str = "cinestill_800t"
    amount: float = 1.0
    impact: float = 0.85
    anti_halation: float = 0.75
    source_selectivity: float = 0.45
    diffusion: float = 0.55
    warm_core: float = 0.45
    background_visibility: float = 0.75
    source_normalization: str = "percentile"


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


def resolve_physical_halation_controls(controls: PhysicalHalationControls) -> dict[str, float | str]:
    """Resolve locked user controls to ``physical_halation_layer`` kwargs.

    Locked invariants:
    - ``amount`` maps only to ``amplify``.
    - ``impact`` maps only to display ``impact``.
    - ``anti_halation`` maps only to backscatter coupling (``no_remjet``).
    - ``diffusion`` maps only to local/global diffusion radii.
    - ``source_selectivity`` maps only to source limiter threshold.
    - ``background_visibility`` maps only to dark-background gating.
    """

    profile = controls.profile
    defaults = _profile_defaults(profile)
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

    return {
        "profile": profile,
        "source_normalization": controls.source_normalization,
        "amplify": amount,
        "impact": impact,
        "source_limiter_stops": _lerp(1.45, 3.05, source_selectivity),
        "source_softness": defaults["softness"],
        "source_gamma": defaults["gamma"],
        "local_diffusion": local_diffusion,
        "global_diffusion": global_diffusion,
        "hue_green": _lerp(defaults["green_min"], defaults["green_max"], warm_core),
        "background_gain": _lerp(0.85, 1.85, background_visibility),
        "background_luma_target": _lerp(0.14, 0.24, background_visibility),
        "no_remjet": _lerp(defaults["anti_halation_min"], defaults["anti_halation_max"], anti_halation),
        "output_alpha_cap": defaults["alpha_cap"],
    }
