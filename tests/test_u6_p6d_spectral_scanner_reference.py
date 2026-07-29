from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_spectral_scanner_reference import (
    evaluate_spectral_scanner_reference,
    load_contract,
    write_report,
)
from src.film_physics.spectral_scanner import (
    SpectralScannerProfile,
    apply_spectral_scanner,
    construct_primary_metamer_pair,
    synthetic_profile_from_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6d_spectral_scanner_reference_v1.json"


def _profiles() -> tuple[SpectralScannerProfile, SpectralScannerProfile]:
    contract = load_contract(CONTRACT)
    grid = contract["wavelength_grid_nm"]
    wavelength = np.arange(
        float(grid["start"]),
        float(grid["stop"]) + float(grid["step"]) * 0.5,
        float(grid["step"]),
        dtype=np.float64,
    )
    return tuple(
        synthetic_profile_from_contract(wavelength, row)
        for row in contract["synthetic_profiles"].values()
    )


def test_clear_and_neutral_spectra_are_normalized() -> None:
    for profile in _profiles():
        clear = np.ones(len(profile.wavelength_nm), dtype=np.float64)
        neutral = np.full_like(clear, 0.37)
        assert apply_spectral_scanner(clear, profile).tolist() == pytest.approx(
            [1.0, 1.0, 1.0], abs=1e-15
        )
        assert apply_spectral_scanner(neutral, profile).tolist() == pytest.approx(
            [0.37, 0.37, 0.37], abs=1e-15
        )


def test_metamer_pair_is_primary_equal_and_observer_separated() -> None:
    primary, observer = _profiles()
    first, second = construct_primary_metamer_pair(
        primary,
        observer,
        neutral_transmittance=0.55,
        perturbation_max_abs=0.28,
    )
    assert np.max(
        np.abs(
            apply_spectral_scanner(first, primary)
            - apply_spectral_scanner(second, primary)
        )
    ) <= 1e-12
    assert np.linalg.norm(
        apply_spectral_scanner(first, observer)
        - apply_spectral_scanner(second, observer)
    ) >= 0.02
    assert np.min(first) > 0.0 and np.max(first) <= 1.0
    assert np.min(second) > 0.0 and np.max(second) <= 1.0


def test_reference_rejects_wrong_dtype_shape_and_domain() -> None:
    primary, _ = _profiles()
    size = len(primary.wavelength_nm)
    with pytest.raises(TypeError, match="float64"):
        apply_spectral_scanner(np.ones(size, dtype=np.float32), primary)
    with pytest.raises(ValueError, match="dimension"):
        apply_spectral_scanner(np.ones(size - 1, dtype=np.float64), primary)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        apply_spectral_scanner(np.zeros(size, dtype=np.float64), primary)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        apply_spectral_scanner(np.full(size, 1.01, dtype=np.float64), primary)


def test_profile_rejects_degenerate_sensor_rows() -> None:
    primary, _ = _profiles()
    sensitivity = [list(row) for row in primary.sensor_sensitivity_rgb]
    sensitivity[1] = sensitivity[0]
    with pytest.raises(ValueError, match="independent"):
        SpectralScannerProfile(
            primary.profile_id,
            primary.wavelength_nm,
            primary.illuminant_spd,
            tuple(tuple(row) for row in sensitivity),
        )


def test_full_report_passes_and_is_byte_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_spectral_scanner_reference(contract)
    second = evaluate_spectral_scanner_reference(contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["metrics"]["rgb_only_irreducible_l2_error"] >= 0.01
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_mutating_observer_to_primary_closes_separation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["synthetic_profiles"]["scanner_b"] = json.loads(
        json.dumps(contract["synthetic_profiles"]["scanner_a"])
    )
    contract["synthetic_profiles"]["scanner_b"]["profile_id"] = (
        "synthetic-spectral-scanner-b-mutated"
    )
    with pytest.raises(ValueError, match="no primary-metamer separation"):
        evaluate_spectral_scanner_reference(contract)
