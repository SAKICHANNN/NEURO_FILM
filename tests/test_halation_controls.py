from __future__ import annotations

from src.filmfx import PhysicalHalationControls, resolve_physical_halation_controls


def changed_keys(left: dict, right: dict) -> set[str]:
    return {key for key in left if left[key] != right[key]}


def test_locked_amount_does_not_change_radius_or_gates() -> None:
    base = resolve_physical_halation_controls(PhysicalHalationControls(amount=0.8))
    stronger = resolve_physical_halation_controls(PhysicalHalationControls(amount=1.5))

    assert changed_keys(base, stronger) == {"amplify"}


def test_locked_impact_only_changes_display_mix() -> None:
    base = resolve_physical_halation_controls(PhysicalHalationControls(impact=0.4))
    stronger = resolve_physical_halation_controls(PhysicalHalationControls(impact=0.9))

    assert changed_keys(base, stronger) == {"impact"}


def test_locked_diffusion_only_changes_geometry_terms() -> None:
    base = resolve_physical_halation_controls(PhysicalHalationControls(diffusion=0.25))
    wider = resolve_physical_halation_controls(PhysicalHalationControls(diffusion=0.75))

    assert changed_keys(base, wider) == {"local_diffusion", "global_diffusion"}
    assert wider["local_diffusion"] > base["local_diffusion"]
    assert wider["global_diffusion"] > base["global_diffusion"]


def test_locked_anti_halation_only_changes_backscatter_coupling() -> None:
    base = resolve_physical_halation_controls(PhysicalHalationControls(anti_halation=0.2))
    no_remjet_like = resolve_physical_halation_controls(PhysicalHalationControls(anti_halation=0.9))

    assert changed_keys(base, no_remjet_like) == {"no_remjet"}
    assert no_remjet_like["no_remjet"] > base["no_remjet"]


def test_locked_source_and_background_sliders_stay_separate() -> None:
    base = resolve_physical_halation_controls(
        PhysicalHalationControls(source_selectivity=0.4, background_visibility=0.6)
    )
    selective = resolve_physical_halation_controls(
        PhysicalHalationControls(source_selectivity=0.8, background_visibility=0.6)
    )
    background = resolve_physical_halation_controls(
        PhysicalHalationControls(source_selectivity=0.4, background_visibility=0.9)
    )

    assert changed_keys(base, selective) == {"source_limiter_stops"}
    assert changed_keys(base, background) == {"background_gain", "background_luma_target"}
    assert selective["source_limiter_stops"] > base["source_limiter_stops"]
    assert background["background_gain"] > base["background_gain"]
    assert background["background_luma_target"] > base["background_luma_target"]


def test_color_response_changes_color_law_not_geometry() -> None:
    deep_red = resolve_physical_halation_controls(
        PhysicalHalationControls(halation_type="cinestill_no_remjet", color_response="deep_red")
    )
    amber = resolve_physical_halation_controls(
        PhysicalHalationControls(halation_type="cinestill_no_remjet", color_response="amber_core")
    )

    assert changed_keys(deep_red, amber) == {"hue_green"}
    assert amber["hue_green"] > deep_red["hue_green"]


def test_bw_density_family_uses_neutral_rule_surface() -> None:
    resolved = resolve_physical_halation_controls(
        PhysicalHalationControls(halation_type="bw_clear_base", color_response="neutral_density")
    )

    assert "density_tint" in resolved
    assert "hue_green" not in resolved
    assert "no_remjet" not in resolved
    assert "profile" not in resolved
