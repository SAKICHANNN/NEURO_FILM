from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.oklab_analytical_interior_confirmation import (
    AnalyticalInteriorConfirmationError,
    _checks,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c11_oklab_analytical_interior_v1.json"


def test_contract_freezes_new_mechanism_without_fit_or_clipping() -> None:
    contract = load_contract(CONTRACT)
    assert contract["experiment_id"] == "U1.4C11"
    assert contract["mapper"]["softness"] == 1.0 / 64.0
    assert contract["mapper"]["rgb16_margin"] == 2.0 / 65535.0
    assert contract["mapper"]["map_only_out_of_gamut_pixels"] is True
    assert contract["mapper"]["cohort_fitting_allowed"] is False
    assert contract["mapper"]["hard_component_clipping_allowed"] is False


def test_contract_rejects_hash_drift(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_bytes(CONTRACT.read_bytes() + b" ")
    with pytest.raises(AnalyticalInteriorConfirmationError, match="contract hash drift"):
        load_contract(changed)


def test_checks_fail_closed_on_boundary_hue_and_nonfinite() -> None:
    contract = load_contract(CONTRACT)
    gates = contract["automatic_gates"]
    metrics = {
        "in_gamut_input_change_count": 0,
        "mapped_output_minimum": 0.0,
        "mapped_output_maximum": 1.0,
        "maximum_new_rgb16_boundary_fraction_vs_semantic_source": 0.0,
        "maximum_per_source_p99_oklab_hue_error_degrees": 0.0,
        "minimum_per_source_median_chroma_scale": 1.0,
        "maximum_per_source_fraction_chroma_scale_below_1e_6": 0.0,
        "population_median_style_retention_ratio": 1.0,
        "worst_render_style_retention_ratio": 1.0,
        "population_median_residual_scale": 1.0,
        "population_median_fraction_scale_below_0p5": 0.0,
        "maximum_new_rgb16_boundary_fraction_vs_mapped_source": 0.0,
        "maximum_p999_gradient_ratio_vs_mapped_source": 1.0,
        "maximum_adjacent_lstar_sign_inversion_fraction": 0.0,
    }
    assert all(_checks(metrics, gates).values())
    metrics["maximum_new_rgb16_boundary_fraction_vs_semantic_source"] = 1e-9
    assert _checks(metrics, gates)["new_mapped_boundary"] is False
    metrics["maximum_new_rgb16_boundary_fraction_vs_semantic_source"] = 0.0
    metrics["maximum_per_source_p99_oklab_hue_error_degrees"] = 0.02
    assert _checks(metrics, gates)["hue_direction"] is False
    metrics["maximum_per_source_p99_oklab_hue_error_degrees"] = float("nan")
    assert _checks(metrics, gates)["finite"] is False
