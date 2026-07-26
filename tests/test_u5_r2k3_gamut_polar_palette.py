from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.roll2film.gamut_polar_palette import (
    GamutPolarPaletteOperator,
    finite_difference_jacobians,
    operator_from_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/u5_r2k3_gamut_polar_palette_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_identity_is_exact_on_boundaries_and_interior() -> None:
    config = _config()
    operator = operator_from_config(
        config["candidate"], config["witnesses"]["identity"]
    )
    rgb = np.vstack(
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.random.default_rng(260726).random((40, 3)),
        )
    )
    np.testing.assert_allclose(operator.apply(rgb), rgb, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(operator.inverse(rgb), rgb, atol=1e-15, rtol=0.0)


def test_nontrivial_witness_is_bounded_and_invertible() -> None:
    config = _config()
    operator = operator_from_config(
        config["candidate"],
        config["witnesses"]["cyan_shadow_warm_highlight_like"],
    )
    axis = np.linspace(0.0, 1.0, 9)
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    output = operator.apply(rgb)
    assert np.min(output) >= -2e-15
    assert np.max(output) <= 1.0 + 2e-15
    np.testing.assert_allclose(operator.inverse(output), rgb, atol=2e-12)


def test_nontrivial_witness_has_positive_interior_jacobian() -> None:
    config = _config()
    operator = operator_from_config(
        config["candidate"], config["witnesses"]["warm_dense_like"]
    )
    axis = np.linspace(0.08, 0.92, 5)
    points = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    determinants = np.linalg.det(
        finite_difference_jacobians(operator, points, step=1e-6)
    )
    assert np.min(determinants) > 0.0


def test_serialization_and_partition_are_exact() -> None:
    config = _config()
    operator = operator_from_config(
        config["candidate"], config["witnesses"]["cross_palette_like"]
    )
    replay = GamutPolarPaletteOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    rgb = np.random.default_rng(17).random((57, 3))
    expected = operator.apply(rgb)
    assert np.array_equal(replay.apply(rgb), expected)
    assert np.array_equal(
        np.concatenate(
            [
                operator.apply(rgb[:13]),
                operator.apply(rgb[13:39]),
                operator.apply(rgb[39:]),
            ]
        ),
        expected,
    )
