from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.velvia_datasheet_witness import (
    SpectralContext,
    build_curve_bank,
    delta_e76,
    film_discriminating_metamer_direction,
    metamer_alternatives,
    metamer_extremes,
    reconstruct_reflectances,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_digitized_curve_bank_is_finite_and_has_frozen_grid() -> None:
    config = _json(ROOT / "configs/u5_r2h0a_velvia_datasheet_witness_v1.json")
    data = _json(ROOT / "configs/data/velvia50_datasheet_curve_pixels_v1.json")
    bank = build_curve_bank(config, data)
    assert bank.wavelength_nm.shape == (69,)
    assert bank.wavelength_nm[0] == 380
    assert bank.wavelength_nm[-1] == 720
    for value in (*bank.sensitivity.values(), *bank.dye_density.values()):
        assert value.shape == (69,)
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0)


def test_smooth_reflectance_solver_reconstructs_feasible_targets() -> None:
    wavelength = 12
    observer = np.array(
        [
            np.linspace(0.02, 0.16, wavelength),
            np.linspace(0.15, 0.03, wavelength),
            0.08 + 0.03 * np.sin(np.linspace(0, np.pi, wavelength)),
        ]
    )
    truth = np.vstack(
        [np.zeros(wavelength), np.full(wavelength, 0.25), np.full(wavelength, 0.8)]
    )
    target = truth @ observer.T
    solved = reconstruct_reflectances(target, observer)
    assert np.min(solved) >= 0
    assert np.max(solved) <= 1
    assert np.max(np.abs(solved @ observer.T - target)) < 1e-6


def test_metamer_direction_is_observer_null_and_alternatives_are_bounded() -> None:
    wavelength = 10
    observer = np.array(
        [
            np.linspace(0.02, 0.12, wavelength),
            np.linspace(0.12, 0.02, wavelength),
            0.06 + 0.02 * np.cos(np.linspace(0, np.pi, wavelength)),
        ]
    )
    film = np.array(
        [
            np.exp(-0.5 * ((np.arange(wavelength) - 2) / 1.5) ** 2),
            np.exp(-0.5 * ((np.arange(wavelength) - 5) / 1.5) ** 2),
            np.exp(-0.5 * ((np.arange(wavelength) - 8) / 1.5) ** 2),
        ]
    )
    context = SpectralContext(
        wavelength_nm=np.arange(wavelength),
        cmf=np.zeros((wavelength, 3)),
        d65=np.ones(wavelength),
        d50=np.ones(wavelength),
        xyz_from_reflectance_d65=observer,
        film_response=film,
        layer_gain=np.ones(3),
    )
    direction = film_discriminating_metamer_direction(context)
    assert np.max(np.abs(observer @ direction)) < 1e-10
    base = np.vstack([np.full(wavelength, 0.2), np.full(wavelength, 0.5)])
    positive, negative = metamer_alternatives(base, direction)
    assert np.min(positive) >= 0 and np.max(positive) <= 1
    assert np.min(negative) >= 0 and np.max(negative) <= 1
    assert np.max(np.abs((positive - base) @ observer.T)) < 1e-10
    assert np.max(np.abs((negative - base) @ observer.T)) < 1e-10


def test_delta_e_is_zero_for_identity() -> None:
    xyz = np.array([[0.2, 0.3, 0.4], [0.95, 1.0, 1.08]])
    np.testing.assert_array_equal(delta_e76(xyz, xyz), np.zeros(2))


def test_per_colour_metamer_extremes_match_observer_and_are_distinct() -> None:
    observer = np.array(
        [
            [0.4, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.4, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.2, 0.4, 0.3],
        ]
    )
    base = np.array([[0.2, 0.3, 0.4, 0.3, 0.2, 0.1]])
    objective = np.array([1.0, -1.0, 0.5, -0.5, 0.8, -0.8])
    high, low = metamer_extremes(base, observer, objective)
    np.testing.assert_allclose(high @ observer.T, base @ observer.T, atol=1e-9)
    np.testing.assert_allclose(low @ observer.T, base @ observer.T, atol=1e-9)
    assert np.linalg.norm(high - low) > 0.1
