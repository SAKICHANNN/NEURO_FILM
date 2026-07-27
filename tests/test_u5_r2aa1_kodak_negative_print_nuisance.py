from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.kodak_negative_print_nuisance import (
    build_curve_bank,
    build_spectral_context,
    negative_transmittance,
    render_chain,
    validate_curve_evidence,
    validate_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict]:
    return (
        _json(ROOT / "configs/u5_r2aa1_kodak_negative_print_nuisance_v1.json"),
        _json(ROOT / "configs/data/kodak_250d_2383_curve_pixels_v1.json"),
    )


def test_curve_bank_has_frozen_support_and_required_families() -> None:
    config, data = _inputs()
    bank = build_curve_bank(config, data)
    np.testing.assert_array_equal(bank.wavelength_nm, np.arange(380, 721, 5))
    assert set(bank.negative_sensitivity) == {
        "yellow_forming",
        "magenta_forming",
        "cyan_forming",
    }
    assert set(bank.print_sensitivity) == set(bank.negative_sensitivity)
    assert set(bank.negative_dyes) == {"yellow", "magenta", "cyan"}
    assert set(bank.print_dyes) == set(bank.negative_dyes)
    for value in (
        *bank.negative_sensitivity.values(),
        *bank.print_sensitivity.values(),
        *bank.negative_dyes.values(),
        *bank.print_dyes.values(),
        bank.negative_midscale,
        bank.negative_dmin,
        bank.print_visual_neutral,
    ):
        assert value.shape == (69,)
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0.0)


def test_curve_evidence_is_on_masked_source_ink() -> None:
    config, data = _inputs()
    evidence = validate_curve_evidence(ROOT, config, data)
    assert max(evidence["axis_max_residual_px"].values()) <= 2.0
    assert max(
        value
        for family in evidence["annotation_max_ink_distance_px"].values()
        for value in family.values()
    ) <= 2.0


def test_all_frozen_source_hashes_and_aa0_branch_match() -> None:
    config, _ = _inputs()
    validate_sources(ROOT, config)


def test_negative_mapping_is_finite_bounded_and_not_identity() -> None:
    config, data = _inputs()
    bank = build_curve_bank(config, data)
    context = build_spectral_context(ROOT, config, bank)
    spectra = np.vstack(
        [
            np.full(69, 0.18),
            np.linspace(0.05, 0.8, 69),
            np.linspace(0.8, 0.05, 69),
        ]
    )
    first = negative_transmittance(
        spectra, context, bank, 2.5, "diagonal_peak_normalized"
    )
    second = negative_transmittance(
        spectra, context, bank, 2.5, "midscale_nnls_scaled"
    )
    for result in (first, second):
        assert result.shape == spectra.shape
        assert np.all(np.isfinite(result))
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
    assert not np.allclose(first, second)


def test_all_preregistered_chain_axes_render_without_clipping() -> None:
    config, data = _inputs()
    bank = build_curve_bank(config, data)
    context = build_spectral_context(ROOT, config, bank)
    reflectance = np.vstack(
        [np.full(69, 0.18), np.linspace(0.1, 0.7, 69)]
    )
    for placement in config["chain"]["negative_neutral_placements"]:
        for mapping in config["chain"]["negative_dye_mappings"]:
            for printer in config["chain"]["printer_hypotheses"]:
                for viewer in config["chain"]["viewing_illuminants"]:
                    result = render_chain(
                        reflectance,
                        context,
                        bank,
                        config,
                        placement=placement,
                        mapping=mapping,
                        printer=printer,
                        viewer=viewer,
                    )
                    assert result["xyz"].shape == (2, 3)
                    assert np.all(np.isfinite(result["xyz"]))
                    assert np.all(np.isfinite(result["linear_srgb"]))
                    assert np.min(result["transmittance"]) >= 0.0
                    assert np.max(result["transmittance"]) <= 1.0
